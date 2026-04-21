#include "serialization/Json.hpp"
#include "models/Bagged.hpp"
#include "models/ClassificationForest.hpp"
#include "models/ClassificationTree.hpp"
#include "models/Projector.hpp"

#include "models/RegressionForest.hpp"
#include "models/RegressionTree.hpp"
#include "models/TrainingSpec.hpp"
#include "models/VariableImportance.hpp"
#include "serialization/ExportValidation.hpp"
#include "serialization/JsonOptional.hpp"
#include "stats/ConfusionMatrix.hpp"
#include "stats/RegressionMetrics.hpp"
#include "stats/Stats.hpp"
#include "utils/Invariant.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

using namespace ppforest2::types;
using namespace ppforest2::stats;

namespace ppforest2::serialization {
  std::vector<std::string>
  to_labels(types::GroupIdVector const& predictions, std::vector<std::string> const& group_names) {
    std::vector<std::string> labels;
    labels.reserve(static_cast<std::size_t>(predictions.size()));

    for (int i = 0; i < predictions.size(); ++i) {
      labels.push_back(group_names[static_cast<std::size_t>(predictions(i))]);
    }

    return labels;
  }

  template<> json Export<Model::Ptr>::to_json() const {
    bool const is_regression = model->training_spec && model->training_spec->mode == types::Mode::Regression;

    // For regression, groups is empty — use the unlabeled serialization path
    // to avoid looking up leaf values in an empty group name vector.
    json result = is_regression ? serialization::to_json(*model) : serialization::to_json(*model, groups);

    result["config"] = model->training_spec->to_json();

    json meta;
    meta["observations"]  = n_observations;
    meta["features"]      = n_features;
    // Centralised `Mode` ↔ string mapping lives in `types::to_string`;
    // keeps this writer and the readers in `TrainingSpec.cpp` /
    // `ExportValidation.cpp` on the same canonical strings.
    meta["mode"]          = types::to_string(is_regression ? types::Mode::Regression : types::Mode::Classification);
    if (!is_regression) {
      meta["groups"] = groups;
    }
    meta["feature_names"] = feature_names;
    result["meta"]        = meta;

    // Optional fields: always present in the output JSON, carrying
    // either the serialized value or `null`. Uniform `nullopt ↔ null`
    // round-trip makes "computed but empty" vs "never computed"
    // distinguishable only via the top-level presence of the key, and
    // removes the hand-editing footgun where a writer omitting a field
    // for a bug looks identical to the legitimate no-data case. The
    // `JsonOptional.hpp` adapter handles plain-typed optionals (e.g.
    // `std::optional<double>`) directly; helpers that take extra args
    // (groups for confusion matrices) use an explicit null fallback.
    result["training_confusion_matrix"] = training_confusion_matrix
        ? serialization::to_json(*training_confusion_matrix, groups)
        : json(nullptr);

    result["oob_error"] = oob_error;

    result["oob_confusion_matrix"] = oob_confusion_matrix
        ? serialization::to_json(*oob_confusion_matrix, groups)
        : json(nullptr);

    result["training_regression_metrics"] = training_regression_metrics
        ? serialization::to_json(*training_regression_metrics)
        : json(nullptr);

    result["oob_regression_metrics"] = oob_regression_metrics
        ? serialization::to_json(*oob_regression_metrics)
        : json(nullptr);

    result["variable_importance"] = variable_importance
        ? serialization::to_json(*variable_importance)
        : json(nullptr);

    return result;
  }

  void JsonNodeVisitor::visit(TreeBranch const& node) {
    JsonNodeVisitor lower_visitor;
    lower_visitor.group_names = group_names;
    node.lower->accept(lower_visitor);

    JsonNodeVisitor upper_visitor;
    upper_visitor.group_names = group_names;
    node.upper->accept(upper_visitor);

    result = json{
        {"projector", node.projector},
        {"cutpoint", node.cutpoint},
        {"pp_index_value", node.pp_index_value},
        {"lower", lower_visitor.result},
        {"upper", upper_visitor.result},
        {"degenerate", node.degenerate},
    };

    if (group_names) {
      std::vector<std::string> named_groups;
      for (auto g : node.groups) {
        named_groups.push_back((*group_names)[static_cast<std::size_t>(g)]);
      }

      result["groups"] = named_groups;
    } else {
      result["groups"] = node.groups;
    }
  }

  void JsonNodeVisitor::visit(TreeLeaf const& node) {
    if (group_names) {
      result = json{
          {"value", (*group_names)[static_cast<std::size_t>(node.value)]},
          {"degenerate", node.degenerate},
      };
    } else {
      result = json{
          {"value", node.value},
          {"degenerate", node.degenerate},
      };
    }
  }

  void JsonModelVisitor::visit(Tree const& tree) {
    result = group_names ? json{{"model_type", "tree"}, {"model", to_json(tree, *group_names)}}
                         : json{{"model_type", "tree"}, {"model", to_json(tree)}};
  }

  void JsonModelVisitor::visit(Forest const& forest) {
    result = group_names ? json{{"model_type", "forest"}, {"model", to_json(forest, *group_names)}}
                         : json{{"model_type", "forest"}, {"model", to_json(forest)}};
  }

  json to_json(Model const& model) {
    JsonModelVisitor visitor;
    model.accept(visitor);
    return visitor.result;
  }

  json to_json(Model const& model, GroupNames const& group_names) {
    JsonModelVisitor visitor;
    visitor.group_names = &group_names;
    model.accept(visitor);
    return visitor.result;
  }

  json to_json(TreeNode const& node) {
    JsonNodeVisitor visitor;
    node.accept(visitor);
    return visitor.result;
  }

  json to_json(Tree const& tree) {
    return json{
        {"root", to_json(*tree.root)},
        {"degenerate", tree.degenerate},
    };
  }

  json to_json(BaggedTree const& tree) {
    json result              = to_json(*tree.model);
    result["sample_indices"] = tree.sample_indices;
    return result;
  }

  json to_json(Forest const& forest) {
    std::vector<json> trees_json;
    trees_json.reserve(forest.trees.size());

    for (auto const& tree : forest.trees) {
      trees_json.push_back(to_json(*tree));
    }

    return json{
        {"trees", trees_json},
        {"degenerate", forest.degenerate},
    };
  }

  json to_json(ConfusionMatrix const& cm) {
    json j;

    std::vector<std::vector<int>> matrix_data;
    for (int i = 0; i < cm.values.rows(); ++i) {
      std::vector<int> row;
      row.reserve(static_cast<std::size_t>(cm.values.cols()));
      for (int col = 0; col < cm.values.cols(); ++col) {
        row.emplace_back(cm.values(i, col));
      }

      matrix_data.push_back(row);
    }

    j["matrix"] = matrix_data;

    std::vector<int> labels;
    labels.reserve(static_cast<std::size_t>(cm.label_index.size()));
    for (auto const& [label, idx] : cm.label_index) {
      labels.push_back(label);
    }

    j["labels"] = labels;

    auto ce = cm.group_errors();
    std::vector<Feature> ce_vec(ce.data(), ce.data() + ce.size());
    j["group_errors"] = ce_vec;

    return j;
  }

  json to_json(VariableImportance const& vi) {
    int const p = static_cast<int>(vi.projections.size());

    std::vector<Feature> scale_vec(vi.scale.data(), vi.scale.data() + p);
    std::vector<Feature> proj_vec(vi.projections.data(), vi.projections.data() + p);

    json j;
    j["scale"]       = scale_vec;
    j["projections"] = proj_vec;

    if (vi.weighted_projections.size() == p) {
      std::vector<Feature> wp_vec(vi.weighted_projections.data(), vi.weighted_projections.data() + p);
      j["weighted_projections"] = wp_vec;
    }

    if (vi.permuted.size() == p) {
      std::vector<Feature> perm_vec(vi.permuted.data(), vi.permuted.data() + p);
      j["permuted"] = perm_vec;
    }

    return j;
  }

  json to_json(RegressionMetrics const& rm) {
    return json{
        {"mse", rm.mse},
        {"mae", rm.mae},
        {"r_squared", rm.r_squared},
    };
  }

  json to_json(TreeNode const& node, GroupNames const& group_names) {
    JsonNodeVisitor visitor;
    visitor.group_names = &group_names;
    node.accept(visitor);
    return visitor.result;
  }

  json to_json(Tree const& tree, GroupNames const& group_names) {
    return json{
        {"root", to_json(*tree.root, group_names)},
        {"degenerate", tree.degenerate},
    };
  }

  json to_json(BaggedTree const& tree, GroupNames const& group_names) {
    json result              = to_json(*tree.model, group_names);
    result["sample_indices"] = tree.sample_indices;
    return result;
  }

  json to_json(Forest const& forest, GroupNames const& group_names) {
    std::vector<json> trees_json;
    trees_json.reserve(forest.trees.size());

    for (auto const& tree : forest.trees) {
      trees_json.push_back(to_json(*tree, group_names));
    }

    return json{
        {"trees", trees_json},
        {"degenerate", forest.degenerate},
    };
  }

  json to_json(ConfusionMatrix const& cm, GroupNames const& group_names) {
    json j;

    std::vector<std::vector<int>> matrix_data;
    for (int i = 0; i < cm.values.rows(); ++i) {
      std::vector<int> row;
      row.reserve(static_cast<std::size_t>(cm.values.cols()));
      for (int col = 0; col < cm.values.cols(); ++col) {
        row.emplace_back(cm.values(i, col));
      }

      matrix_data.push_back(row);
    }

    j["matrix"] = matrix_data;

    std::vector<std::string> labels;
    labels.reserve(static_cast<std::size_t>(cm.label_index.size()));
    for (auto const& [label, idx] : cm.label_index) {
      labels.emplace_back(group_names[static_cast<std::size_t>(label)]);
    }

    j["labels"] = labels;

    auto ce = cm.group_errors();
    std::vector<Feature> ce_vec(ce.data(), ce.data() + ce.size());
    j["group_errors"] = ce_vec;

    return j;
  }

  json to_json(FeatureMatrix const& matrix) {
    std::vector<std::vector<Feature>> rows;
    rows.reserve(static_cast<std::size_t>(matrix.rows()));

    for (int i = 0; i < matrix.rows(); ++i) {
      std::vector<Feature> row(static_cast<std::size_t>(matrix.cols()));

      for (int j = 0; j < matrix.cols(); ++j) {
        row[static_cast<std::size_t>(j)] = matrix(i, j);
      }

      rows.push_back(std::move(row));
    }

    return json(rows);
  }

  // -----------------------------------------------------------------------
  // Internal: node deserialization (integer labels only)
  // -----------------------------------------------------------------------

  static TreeNode::Ptr node_from_json(json const& j) {
    if (j.contains("value")) {
      // Outcome is a float (= Feature). Classification leaves hold integer-valued floats;
      // regression leaves hold arbitrary floats. Reading as Outcome handles both.
      auto leaf        = TreeLeaf::make(j["value"].get<Outcome>());
      leaf->degenerate = j.value("degenerate", false);
      return leaf;
    }

    auto const proj_vec = j["projector"].get<std::vector<Feature>>();

    pp::Projector projector = Eigen::Map<pp::Projector const>(proj_vec.data(), static_cast<int>(proj_vec.size()));

    Feature const cut            = j["cutpoint"].get<Feature>();
    Feature const pp_index_value = j.value("pp_index_value", Feature(0));

    std::set<GroupId> groups;

    if (j.contains("groups")) {
      for (auto const& g : j["groups"]) {
        groups.insert(g.get<GroupId>());
      }
    }

    auto node = TreeBranch::make(
        projector, cut, node_from_json(j["lower"]), node_from_json(j["upper"]), groups, pp_index_value
    );

    node->degenerate = node->degenerate || j.value("degenerate", false);
    return node;
  }

  // -----------------------------------------------------------------------
  // Internal: node deserialization (labeled — resolves string → int)
  // -----------------------------------------------------------------------

  static GroupId resolve_label(json const& value, GroupNames const& group_names) {
    if (value.is_string()) {
      auto it = std::find(group_names.begin(), group_names.end(), value.get<std::string>());
      return static_cast<GroupId>(std::distance(group_names.begin(), it));
    }

    return value.get<GroupId>();
  }

  static TreeNode::Ptr node_from_json(json const& j, GroupNames const& group_names) {
    if (j.contains("value")) {
      auto leaf        = TreeLeaf::make(resolve_label(j["value"], group_names));
      leaf->degenerate = j.value("degenerate", false);
      return leaf;
    }

    auto const proj_vec = j["projector"].get<std::vector<Feature>>();

    pp::Projector projector = Eigen::Map<pp::Projector const>(proj_vec.data(), static_cast<int>(proj_vec.size()));

    Feature const cut            = j["cutpoint"].get<Feature>();
    Feature const pp_index_value = j.value("pp_index_value", Feature(0));

    std::set<GroupId> groups;

    if (j.contains("groups")) {
      for (auto const& g : j["groups"]) {
        groups.insert(resolve_label(g, group_names));
      }
    }

    auto node = TreeBranch::make(
        projector,
        cut,
        node_from_json(j["lower"], group_names),
        node_from_json(j["upper"], group_names),
        groups,
        pp_index_value
    );

    node->degenerate = node->degenerate || j.value("degenerate", false);
    return node;
  }

  // -----------------------------------------------------------------------
  // Deserialization — bare model block (integer labels, no spec)
  // -----------------------------------------------------------------------

  // Bare model deserialization — no TrainingSpec context available, so we
  // default to the Classification subclasses and synthesize a placeholder
  // classification spec to satisfy the subclass/mode invariant. Callers
  // that need regression semantics (or the original training config) must
  // go through the Export<Tree::Ptr>/Export<Forest::Ptr> path, which
  // rehydrates the real spec from the `config` JSON block.
  namespace {
    TrainingSpec::Ptr placeholder_classification_spec() {
      return TrainingSpec::builder(types::Mode::Classification).make();
    }
  }

  template<> Tree::Ptr from_json<Tree::Ptr>(json const& j) {
    return std::make_unique<ClassificationTree>(
        node_from_json(j["root"]), placeholder_classification_spec()
    );
  }

  template<> Forest::Ptr from_json<Forest::Ptr>(json const& j) {
    auto forest        = std::make_unique<ClassificationForest>(placeholder_classification_spec());
    forest->degenerate = j.value("degenerate", false);

    for (auto const& tree_json : j.at("trees")) {
      auto sample_indices = tree_json.contains("sample_indices")
          ? tree_json["sample_indices"].get<std::vector<int>>()
          : std::vector<int>{};

      std::unique_ptr<Tree> inner_tree = std::make_unique<ClassificationTree>(
          node_from_json(tree_json.at("root")), placeholder_classification_spec()
      );

      forest->add_tree(
          std::make_unique<BaggedTree>(std::move(inner_tree), std::move(sample_indices))
      );
    }

    return forest;
  }

  template<> ConfusionMatrix from_json<ConfusionMatrix>(json const& j) {
    ConfusionMatrix cm;
    auto const& matrix_data = j["matrix"];
    int n                   = static_cast<int>(matrix_data.size());
    cm.values               = types::Matrix<int>::Zero(n, n);

    for (int i = 0; i < n; ++i) {
      for (int col = 0; col < n; ++col) {
        cm.values(i, col) = matrix_data[static_cast<std::size_t>(i)][static_cast<std::size_t>(col)].get<int>();
      }
    }

    auto const& labels = j["labels"];

    for (int i = 0; i < n; ++i) {
      if (labels[static_cast<std::size_t>(i)].is_number_integer()) {
        cm.label_index[labels[static_cast<std::size_t>(i)].get<int>()] = i;
      } else {
        cm.label_index[i] = i;
      }
    }

    return cm;
  }

  template<> VariableImportance from_json<VariableImportance>(json const& j) {
    VariableImportance vi;
    auto scale_vec = j["scale"].get<std::vector<Feature>>();
    auto proj_vec  = j["projections"].get<std::vector<Feature>>();
    int p          = static_cast<int>(proj_vec.size());

    vi.scale       = Eigen::Map<types::FeatureVector const>(scale_vec.data(), p);
    vi.projections = Eigen::Map<types::FeatureVector const>(proj_vec.data(), p);

    if (j.contains("weighted_projections") && !j["weighted_projections"].empty()) {
      auto wp_vec             = j["weighted_projections"].get<std::vector<Feature>>();
      vi.weighted_projections = Eigen::Map<types::FeatureVector const>(wp_vec.data(), p);
    }

    if (j.contains("permuted") && !j["permuted"].empty()) {
      auto perm_vec = j["permuted"].get<std::vector<Feature>>();
      vi.permuted   = Eigen::Map<types::FeatureVector const>(perm_vec.data(), p);
    }

    return vi;
  }

  template<> RegressionMetrics from_json<RegressionMetrics>(json const& j) {
    RegressionMetrics rm;
    rm.mse       = j.at("mse").get<double>();
    rm.mae       = j.at("mae").get<double>();
    rm.r_squared = j.at("r_squared").get<double>();
    return rm;
  }

  template<> void Export<Model::Ptr>::compute_metrics(
      types::FeatureMatrix const& x,
      types::OutcomeVector const& y
  ) {
    int const seed           = model->training_spec->seed;
    bool const is_regression = model->training_spec->mode == types::Mode::Regression;

    // Training metrics
    OutcomeVector train_preds = model->predict(x);

    if (is_regression) {
      training_regression_metrics = RegressionMetrics(train_preds, y);
    } else {
      GroupIdVector train_preds_int = train_preds.cast<GroupId>();
      GroupIdVector y_int           = y.cast<GroupId>();
      training_confusion_matrix     = ConfusionMatrix(train_preds_int, y_int);
    }

    // Model-specific metrics (OOB for forests)
    struct MetricsVisitor : Model::Visitor {
      FeatureMatrix const& x;
      OutcomeVector const& y;
      int seed;
      bool is_regression;
      Export<Model::Ptr>& self;

      MetricsVisitor(
          FeatureMatrix const& x,
          OutcomeVector const& y,
          int seed,
          bool is_regression,
          Export<Model::Ptr>& self
      )
          : x(x)
          , y(y)
          , seed(seed)
          , is_regression(is_regression)
          , self(self) {}

      void visit(Forest const& forest) override {
        OutcomeVector oob_preds = forest.oob_predict(x);

        // "No OOB tree" sentinel: NaN for regression, -1 for classification.
        std::vector<int> oob_rows;
        for (int i = 0; i < oob_preds.size(); ++i) {
          bool const no_oob = is_regression
              ? std::isnan(static_cast<double>(oob_preds(i)))
              : oob_preds(i) == Outcome(-1);

          if (!no_oob) {
            oob_rows.emplace_back(i);
          }
        }

        if (!oob_rows.empty()) {
          OutcomeVector preds_oob = oob_preds(oob_rows, Eigen::all).eval();

          if (is_regression) {
            OutcomeVector y_oob         = y(oob_rows, Eigen::all).eval();
            self.oob_regression_metrics = RegressionMetrics(preds_oob, y_oob);
            self.oob_error              = self.oob_regression_metrics->mse;
          } else {
            // Subset first, cast second — avoids materializing a full-size
            // int vector just to read a small OOB slice per tree.
            GroupIdVector y_oob_int     = y(oob_rows, Eigen::all).cast<GroupId>().eval();
            self.oob_error              = error_rate(preds_oob, y_oob_int);
            GroupIdVector preds_oob_int = preds_oob.cast<GroupId>();
            self.oob_confusion_matrix   = ConfusionMatrix(preds_oob_int, y_oob_int);
          }
        }

        // Unified-y: `y` already carries the right values for both modes
        // (continuous response for regression; float-encoded class labels for
        // classification). The mode-specific VI overrides handle the cast.
        self.variable_importance = forest.variable_importance(x, y, seed);
      }

      void visit(Tree const& tree) override {
        self.variable_importance = tree.variable_importance(x);
      }
    };

    MetricsVisitor visitor(x, y, seed, is_regression, *this);
    model->accept(visitor);
  }
}

// -------------------------------------------------------------------------
// adl_serializer implementations — Export<T>
// -------------------------------------------------------------------------
namespace nlohmann {
  using namespace ppforest2;
  using namespace ppforest2::serialization;
  using json = nlohmann::json;

  // Full export (labeled JSON + config + meta) — constructs the concrete
  // subclass (Classification or Regression) based on the spec's mode.
  Export<Tree::Ptr> adl_serializer<Export<Tree::Ptr>>::from_json(json const& j) {
    // Run skeleton/config/meta checks up-front with path-annotated errors.
    // Everything below this point can assume well-formed structure.
    validate_tree_export(j);

    auto spec         = TrainingSpec::from_json(j.at("config"));
    auto const& meta  = j.at("meta");
    GroupNames groups = meta.contains("groups") ? meta["groups"].get<GroupNames>() : GroupNames{};

    bool const is_regression = spec->mode == types::Mode::Regression;

    auto root = is_regression ? node_from_json(j.at("model")["root"])
                              : node_from_json(j.at("model")["root"], groups);

    Tree::Ptr tree = is_regression
        ? static_cast<Tree::Ptr>(std::make_unique<RegressionTree>(std::move(root), spec))
        : static_cast<Tree::Ptr>(std::make_unique<ClassificationTree>(std::move(root), spec));

    return {
        std::move(tree),
        std::move(groups),
        std::move(spec),
        meta.value("observations", 0),
        meta.value("features", 0),
        meta.contains("feature_names") ? meta["feature_names"].get<std::vector<std::string>>()
                                       : std::vector<std::string>{},
    };
  }

  Export<Forest::Ptr> adl_serializer<Export<Forest::Ptr>>::from_json(json const& j) {
    validate_forest_export(j);

    auto spec         = TrainingSpec::from_json(j.at("config"));
    auto const& meta  = j.at("meta");
    GroupNames groups = meta.contains("groups") ? meta["groups"].get<GroupNames>() : GroupNames{};
    auto const& mj    = j.at("model");

    bool const is_regression = spec->mode == types::Mode::Regression;

    Forest::Ptr forest = is_regression
        ? static_cast<Forest::Ptr>(std::make_unique<RegressionForest>(spec))
        : static_cast<Forest::Ptr>(std::make_unique<ClassificationForest>(spec));

    forest->degenerate = mj.value("degenerate", false);

    for (auto const& tree_json : mj.at("trees")) {
      auto sample_indices = tree_json.contains("sample_indices")
          ? tree_json["sample_indices"].get<std::vector<int>>()
          : std::vector<int>{};

      auto root = is_regression ? node_from_json(tree_json.at("root"))
                                : node_from_json(tree_json.at("root"), groups);

      // The bag wrapper is mode-agnostic (`BaggedTree` = `Bagged<Tree>`).
      // Mode lives on the inner `Tree` subclass; the wrapper just holds
      // the pair. `Forest::add_tree` checks the mode match at runtime.
      std::unique_ptr<Tree> inner = is_regression
          ? static_cast<std::unique_ptr<Tree>>(std::make_unique<RegressionTree>(std::move(root), spec))
          : static_cast<std::unique_ptr<Tree>>(std::make_unique<ClassificationTree>(std::move(root), spec));

      forest->add_tree(
          std::make_unique<BaggedTree>(std::move(inner), std::move(sample_indices))
      );
    }

    return {
        std::move(forest),
        std::move(groups),
        std::move(spec),
        meta.value("observations", 0),
        meta.value("features", 0),
        meta.contains("feature_names") ? meta["feature_names"].get<std::vector<std::string>>()
                                       : std::vector<std::string>{},
    };
  }

  Export<Model::Ptr> adl_serializer<Export<Model::Ptr>>::from_json(json const& j) {
    std::string model_type = j.value("model_type", "tree");

    Export<Model::Ptr> result = [&]() -> Export<Model::Ptr> {
      if (model_type == "forest") {
        auto fe = j.get<Export<Forest::Ptr>>();
        return {
            std::shared_ptr<Forest>(fe.model.release()),
            std::move(fe.groups),
            std::move(fe.spec),
            fe.n_observations,
            fe.n_features,
            std::move(fe.feature_names),
        };
      }

      if (model_type == "tree") {
        auto te = j.get<Export<Tree::Ptr>>();
        return {
            std::shared_ptr<Tree>(te.model.release()),
            std::move(te.groups),
            std::move(te.spec),
            te.n_observations,
            te.n_features,
            std::move(te.feature_names),
        };
      }

      throw std::invalid_argument("Invalid model type: " + model_type);
    }();

    // Optional-field reader. Each branch treats "key absent" and
    // "key present but null" identically — both map to `std::nullopt`.
    // The current writer always emits the key (with a value or `null`),
    // but older JSON files or hand-edits may have the key missing
    // entirely; both stay supported via the shared `has_value` helper
    // (same helper used in Presentation.cpp and Summarize.cpp — one
    // convention across the codebase, not two).
    using serialization::has_value;
    if (has_value(j, "variable_importance")) {
      result.variable_importance = serialization::from_json<VariableImportance>(j["variable_importance"]);
    }
    if (has_value(j, "training_confusion_matrix")) {
      result.training_confusion_matrix = serialization::from_json<ConfusionMatrix>(j["training_confusion_matrix"]);
    }
    if (has_value(j, "oob_confusion_matrix")) {
      result.oob_confusion_matrix = serialization::from_json<ConfusionMatrix>(j["oob_confusion_matrix"]);
    }
    if (has_value(j, "oob_error")) {
      result.oob_error = j["oob_error"].get<double>();
    }
    if (has_value(j, "training_regression_metrics")) {
      result.training_regression_metrics = serialization::from_json<RegressionMetrics>(j["training_regression_metrics"]);
    }
    if (has_value(j, "oob_regression_metrics")) {
      result.oob_regression_metrics = serialization::from_json<RegressionMetrics>(j["oob_regression_metrics"]);
    }
    return result;
  }
}

namespace ppforest2::serialization {
  std::ostream& operator<<(std::ostream& os, TreeNode const& node) {
    return os << to_json(node).dump(2, ' ', false);
  }

  std::ostream& operator<<(std::ostream& os, TreeBranch const& condition) {
    return os << to_json(static_cast<TreeNode const&>(condition)).dump(2, ' ', false);
  }

  std::ostream& operator<<(std::ostream& os, TreeLeaf const& response) {
    return os << to_json(static_cast<TreeNode const&>(response)).dump(2, ' ', false);
  }

  std::ostream& operator<<(std::ostream& os, Tree const& tree) {
    return os << to_json(tree).dump(2, ' ', false);
  }

  std::ostream& operator<<(std::ostream& os, Forest const& forest) {
    return os << to_json(forest).dump(2, ' ', false);
  }
}
