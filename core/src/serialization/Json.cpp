#include "serialization/Json.hpp"
#include "models/Projector.hpp"
#include "models/TrainingSpec.hpp"
#include "models/VariableImportance.hpp"
#include "stats/ConfusionMatrix.hpp"
#include "stats/Stats.hpp"

#include <Eigen/Dense>
#include <algorithm>

using namespace ppforest2::types;
using namespace ppforest2::stats;

namespace ppforest2::serialization {
  std::vector<std::string>
  to_labels(types::OutcomeVector const& predictions, std::vector<std::string> const& group_names) {
    std::vector<std::string> labels;
    labels.reserve(static_cast<std::size_t>(predictions.size()));

    for (int i = 0; i < predictions.size(); ++i) {
      labels.push_back(group_names[static_cast<std::size_t>(predictions(i))]);
    }

    return labels;
  }

  template<> json Export<Model::Ptr>::to_json() const {
    json result = serialization::to_json(*model, groups);

    result["config"] = model->training_spec->to_json();

    json meta;
    meta["observations"]  = n_observations;
    meta["features"]      = n_features;
    meta["groups"]        = groups;
    meta["feature_names"] = feature_names;
    result["meta"]        = meta;

    if (training_confusion_matrix)
      result["training_confusion_matrix"] = serialization::to_json(*training_confusion_matrix, groups);

    if (oob_error)
      result["oob_error"] = *oob_error;

    if (oob_confusion_matrix)
      result["oob_confusion_matrix"] = serialization::to_json(*oob_confusion_matrix, groups);

    if (variable_importance)
      result["variable_importance"] = serialization::to_json(*variable_importance);

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

  json to_json(BootstrapTree const& tree) {
    json result              = to_json(static_cast<Tree const&>(tree));
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
    std::vector<float> ce_vec(ce.data(), ce.data() + ce.size());
    j["group_errors"] = ce_vec;

    return j;
  }

  json to_json(VariableImportance const& vi) {
    int const p = static_cast<int>(vi.projections.size());

    std::vector<float> scale_vec(vi.scale.data(), vi.scale.data() + p);
    std::vector<float> proj_vec(vi.projections.data(), vi.projections.data() + p);

    json j;
    j["scale"]       = scale_vec;
    j["projections"] = proj_vec;

    if (vi.weighted_projections.size() == p) {
      std::vector<float> wp_vec(vi.weighted_projections.data(), vi.weighted_projections.data() + p);
      j["weighted_projections"] = wp_vec;
    }

    if (vi.permuted.size() == p) {
      std::vector<float> perm_vec(vi.permuted.data(), vi.permuted.data() + p);
      j["permuted"] = perm_vec;
    }

    return j;
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

  json to_json(BootstrapTree const& tree, GroupNames const& group_names) {
    json result              = to_json(static_cast<Tree const&>(tree), group_names);
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
    std::vector<float> ce_vec(ce.data(), ce.data() + ce.size());
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
      auto leaf        = TreeLeaf::make(j["value"].get<Outcome>());
      leaf->degenerate = j.value("degenerate", false);
      return leaf;
    }

    auto const proj_vec = j["projector"].get<std::vector<Feature>>();

    pp::Projector projector = Eigen::Map<pp::Projector const>(proj_vec.data(), static_cast<int>(proj_vec.size()));

    Feature const cut            = j["cutpoint"].get<Feature>();
    Feature const pp_index_value = j.value("pp_index_value", Feature(0));

    std::set<Outcome> groups;

    if (j.contains("groups")) {
      for (auto const& g : j["groups"]) {
        groups.insert(g.get<Outcome>());
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

  static Outcome resolve_label(json const& value, GroupNames const& group_names) {
    if (value.is_string()) {
      auto it = std::find(group_names.begin(), group_names.end(), value.get<std::string>());
      return static_cast<Outcome>(std::distance(group_names.begin(), it));
    }

    return value.get<Outcome>();
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

    std::set<Outcome> groups;

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

  template<> Tree from_json<Tree>(json const& j) {
    return Tree(node_from_json(j["root"]), nullptr);
  }

  template<> Forest from_json<Forest>(json const& j) {
    Forest forest(nullptr);
    forest.degenerate = j.value("degenerate", false);

    for (auto const& tree_json : j.at("trees")) {
      auto sample_indices = tree_json.contains("sample_indices") ? tree_json["sample_indices"].get<std::vector<int>>()
                                                                 : std::vector<int>{};

      forest.add_tree(
          std::make_unique<BootstrapTree>(node_from_json(tree_json.at("root")), nullptr, std::move(sample_indices))
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
    auto scale_vec = j["scale"].get<std::vector<float>>();
    auto proj_vec  = j["projections"].get<std::vector<float>>();
    int p          = static_cast<int>(proj_vec.size());

    vi.scale       = Eigen::Map<types::FeatureVector const>(scale_vec.data(), p);
    vi.projections = Eigen::Map<types::FeatureVector const>(proj_vec.data(), p);

    if (j.contains("weighted_projections") && !j["weighted_projections"].empty()) {
      auto wp_vec             = j["weighted_projections"].get<std::vector<float>>();
      vi.weighted_projections = Eigen::Map<types::FeatureVector const>(wp_vec.data(), p);
    }

    if (j.contains("permuted") && !j["permuted"].empty()) {
      auto perm_vec = j["permuted"].get<std::vector<float>>();
      vi.permuted   = Eigen::Map<types::FeatureVector const>(perm_vec.data(), p);
    }

    return vi;
  }

  template<> void Export<Model::Ptr>::compute_metrics(types::FeatureMatrix const& x, types::OutcomeVector const& y) {
    int const n_vars = static_cast<int>(x.cols());
    int const seed   = model->training_spec->seed;

    VariableImportance vi;
    vi.scale = sd(x);
    vi.scale = (vi.scale.array() > Feature(0)).select(vi.scale, Feature(1));

    // Training confusion matrix
    OutcomeVector train_preds = model->predict(x);
    training_confusion_matrix = ConfusionMatrix(train_preds, y);

    // Model-specific metrics (OOB for forests)
    struct MetricsVisitor : Model::Visitor {
      FeatureMatrix const& x;
      OutcomeVector const& y;
      int n_vars, seed;
      VariableImportance& vi;
      Export<Model::Ptr>& self;

      MetricsVisitor(
          FeatureMatrix const& x,
          OutcomeVector const& y,
          int n_vars,
          int seed,
          VariableImportance& vi,
          Export<Model::Ptr>& self
      )
          : x(x)
          , y(y)
          , n_vars(n_vars)
          , seed(seed)
          , vi(vi)
          , self(self) {}

      void visit(Forest const& forest) override {
        OutcomeVector oob_preds = forest.oob_predict(x);

        std::vector<int> oob_rows;
        for (int i = 0; i < oob_preds.size(); ++i) {
          if (oob_preds(i) >= 0) {
            oob_rows.emplace_back(i);
          }
        }

        if (!oob_rows.empty()) {
          OutcomeVector preds_oob   = oob_preds(oob_rows, Eigen::all).eval();
          OutcomeVector y_oob       = y(oob_rows, Eigen::all).eval();
          self.oob_error            = error_rate(preds_oob, y_oob);
          self.oob_confusion_matrix = ConfusionMatrix(preds_oob, y_oob);
        }

        vi.permuted             = variable_importance_permuted(forest, x, y, seed);
        vi.projections          = variable_importance_projections(forest, n_vars, &vi.scale);
        vi.weighted_projections = variable_importance_weighted_projections(forest, x, y, &vi.scale);
      }

      void visit(Tree const& tree) override {
        vi.projections = variable_importance_projections(tree, n_vars, &vi.scale);
      }
    };

    MetricsVisitor visitor(x, y, n_vars, seed, vi, *this);
    model->accept(visitor);

    variable_importance = std::move(vi);
  }
}

// -------------------------------------------------------------------------
// adl_serializer implementations — Export<T>
// -------------------------------------------------------------------------
namespace nlohmann {
  using namespace ppforest2;
  using namespace ppforest2::serialization;
  using json = nlohmann::json;

  // Full export (labeled JSON + config + meta)
  Export<Tree> adl_serializer<Export<Tree>>::from_json(json const& j) {
    auto spec        = TrainingSpec::from_json(j.at("config"));
    auto const& meta = j.at("meta");
    auto groups      = meta.at("groups").get<GroupNames>();

    return {
        Tree(node_from_json(j.at("model")["root"], groups), spec),
        std::move(groups),
        std::move(spec),
        meta.value("observations", 0),
        meta.value("features", 0),
        meta.contains("feature_names") ? meta["feature_names"].get<std::vector<std::string>>()
                                       : std::vector<std::string>{},
    };
  }

  Export<Forest> adl_serializer<Export<Forest>>::from_json(json const& j) {
    auto spec        = TrainingSpec::from_json(j.at("config"));
    auto const& meta = j.at("meta");
    auto groups      = meta.at("groups").get<GroupNames>();
    auto const& mj   = j.at("model");

    Forest forest(spec);
    forest.degenerate = mj.value("degenerate", false);

    for (auto const& tree_json : mj.at("trees")) {
      auto sample_indices = tree_json.contains("sample_indices") ? tree_json["sample_indices"].get<std::vector<int>>()
                                                                 : std::vector<int>{};

      forest.add_tree(
          std::make_unique<BootstrapTree>(node_from_json(tree_json.at("root"), groups), spec, std::move(sample_indices))
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
        auto fe = j.get<Export<Forest>>();
        return {
            std::make_shared<Forest>(std::move(fe.model)),
            std::move(fe.groups),
            std::move(fe.spec),
            fe.n_observations,
            fe.n_features,
            std::move(fe.feature_names),
        };
      }

      if (model_type == "tree") {
        auto te = j.get<Export<Tree>>();
        return {
            std::make_shared<Tree>(std::move(te.model)),
            std::move(te.groups),
            std::move(te.spec),
            te.n_observations,
            te.n_features,
            std::move(te.feature_names),
        };
      }

      throw std::invalid_argument("Invalid model type: " + model_type);
    }();

    if (j.contains("variable_importance")) {
      result.variable_importance = serialization::from_json<VariableImportance>(j["variable_importance"]);
    }
    if (j.contains("training_confusion_matrix")) {
      result.training_confusion_matrix = serialization::from_json<ConfusionMatrix>(j["training_confusion_matrix"]);
    }
    if (j.contains("oob_confusion_matrix")) {
      result.oob_confusion_matrix = serialization::from_json<ConfusionMatrix>(j["oob_confusion_matrix"]);
    }
    if (j.contains("oob_error")) {
      result.oob_error = j["oob_error"].get<double>();
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
