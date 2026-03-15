#include <Rcpp.h>
#include <RcppEigen.h>
#include "ppforest2.hpp"
#include "serialization/Json.hpp"

#include <nlohmann/json.hpp>
#include <thread>

// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
using namespace RcppEigen;
using namespace ppforest2;
using namespace ppforest2::types;
using namespace ppforest2::stats;
using namespace ppforest2::viz;

// [[Rcpp::export]]
bool ppforest2_has_openmp() {
  #ifdef _OPENMP
  return true;
  #else
  return false;
  #endif
}

// [[Rcpp::export]]
Tree ppforest2_train_tree_glda(
  FeatureMatrix  x,
  ResponseVector y,
  const float    lambda,
  const int      seed) {
  y.array() -= 1;  // R 1-based → C++ 0-based

  if (!GroupPartition::is_contiguous(y)) {
    sort(x, y);
  }

  RNG rng(seed);

  return Tree::train(
    TrainingSpecGLDA(lambda),
    x,
    y,
    rng);
}

// [[Rcpp::export]]
Forest ppforest2_train_forest_glda(
  FeatureMatrix  x,
  ResponseVector y,
  const int      size,
  const int      n_vars,
  const float    lambda,
  const int      seed,
  SEXP           n_threads,
  const int      max_retries = 3) {
  y.array() -= 1;  // R 1-based → C++ 0-based

  if (!GroupPartition::is_contiguous(y)) {
    sort(x, y);
  }

  if (n_threads == R_NilValue) {
    return Forest::train(
      TrainingSpecUGLDA(n_vars, lambda),
      x,
      y,
      size,
      seed,
      std::thread::hardware_concurrency(),
      max_retries);
  }

  return Forest::train(
    TrainingSpecUGLDA(n_vars, lambda),
    x,
    y,
    size,
    seed,
    as<const int>(n_threads),
    max_retries);
}

// [[Rcpp::export]]
ResponseVector ppforest2_predict(
  const Tree &          tree,
  const FeatureMatrix & data) {
  ResponseVector result = tree.predict(data);
  result.array() += 1;  // C++ 0-based → R 1-based
  return result;
}

// [[Rcpp::export]]
ResponseVector ppforest2_predict_forest(
  const Forest &        forest,
  const FeatureMatrix & data) {
  ResponseVector result = forest.predict(data);
  result.array() += 1;  // C++ 0-based → R 1-based
  return result;
}

// [[Rcpp::export]]
FeatureMatrix ppforest2_predict_forest_prob(
  const Forest &        forest,
  const FeatureMatrix & data) {
  return forest.predict(data, Proportions{});
}

// [[Rcpp::export]]
FeatureVector ppforest2_vi_projections_tree(
  const Tree&          tree,
  int                  n_vars,
  const FeatureVector& scale) {
  return variable_importance_projections(tree, n_vars, &scale);
}

// [[Rcpp::export]]
FeatureVector ppforest2_vi_projections_forest(
  const Forest&        forest,
  int                 n_vars,
  const FeatureVector& scale) {
  return variable_importance_projections(forest, n_vars, &scale);
}

// [[Rcpp::export]]
FeatureVector ppforest2_vi_weighted(
  const Forest&        forest,
  const FeatureMatrix& x,
  ResponseVector       y,
  const FeatureVector& scale) {
  y.array() -= 1;  // R 1-based → C++ 0-based
  return variable_importance_weighted_projections(forest, x, y, &scale);
}

// [[Rcpp::export]]
FeatureVector ppforest2_vi_permuted(
  const Forest&        forest,
  const FeatureMatrix& x,
  ResponseVector       y,
  int                  seed) {
  y.array() -= 1;  // R 1-based → C++ 0-based
  return variable_importance_permuted(forest, x, y, seed);
}

// [[Rcpp::export]]
double ppforest2_oob_error(
  const Forest&        forest,
  const FeatureMatrix& x,
  ResponseVector       y) {
  y.array() -= 1;  // R 1-based → C++ 0-based
  return forest.oob_error(x, y);
}

// [[Rcpp::export]]
Rcpp::List ppforest2_tree_node_data(
  const Tree&          tree,
  const FeatureMatrix& x,
  ResponseVector       y) {
  y.array() -= 1;  // R 1-based → C++ 0-based
  NodeDataVisitor visitor(x, y);
  tree.root->accept(visitor);

  Rcpp::List result(visitor.nodes.size());

  for (std::size_t i = 0; i < visitor.nodes.size(); ++i) {
    const auto& nd = visitor.nodes[i];

    // Convert 0-based class indices → R 1-based
    Rcpp::IntegerVector classes_r(nd.classes.begin(), nd.classes.end());
    classes_r = classes_r + 1;

    if (nd.is_leaf) {
      result[i] = Rcpp::List::create(
        Rcpp::Named("is_leaf")   = true,
        Rcpp::Named("depth")     = nd.depth,
        Rcpp::Named("value")     = nd.value + 1,
        Rcpp::Named("classes")   = classes_r
      );
    } else {
      result[i] = Rcpp::List::create(
        Rcpp::Named("is_leaf")   = false,
        Rcpp::Named("depth")     = nd.depth,
        Rcpp::Named("projector") = Rcpp::wrap(nd.projector),
        Rcpp::Named("threshold") = nd.threshold,
        Rcpp::Named("projected") = Rcpp::NumericVector(nd.projected_values.begin(), nd.projected_values.end()),
        Rcpp::Named("classes")   = classes_r
      );
    }
  }

  return result;
}

// [[Rcpp::export]]
Rcpp::DataFrame ppforest2_boundary_segments(
  const Tree&             tree,
  Rcpp::IntegerVector     var_indices,
  Rcpp::NumericVector     fixed_values,
  double                  x_min,
  double                  x_max,
  double                  y_min,
  double                  y_max) {
  int vi = var_indices[0];
  int vj = var_indices[1];

  int p = static_cast<int>(var_indices.size()) + static_cast<int>(fixed_values.size());

  std::vector<std::pair<int, types::Feature>> fixed_vars;

  if (fixed_values.size() > 0) {
    // Find the variable index not in var_indices
    std::set<int> used(var_indices.begin(), var_indices.end());
    int fv_idx = 0;

    for (int k = 0; k < p; ++k) {
      if (used.find(k) == used.end()) {
        fixed_vars.push_back({ k, static_cast<types::Feature>(fixed_values[fv_idx++]) });
      }
    }
  }

  BoundaryVisitor visitor(
    vi, vj, fixed_vars,
    static_cast<types::Feature>(x_min),
    static_cast<types::Feature>(x_max),
    static_cast<types::Feature>(y_min),
    static_cast<types::Feature>(y_max));

  tree.root->accept(visitor);

  int n = static_cast<int>(visitor.segments.size());
  Rcpp::NumericVector xs(n), ys(n), xe(n), ye(n);
  Rcpp::IntegerVector depths(n);

  for (int i = 0; i < n; ++i) {
    xs[i] = visitor.segments[static_cast<std::size_t>(i)].x_start;
    ys[i] = visitor.segments[static_cast<std::size_t>(i)].y_start;
    xe[i] = visitor.segments[static_cast<std::size_t>(i)].x_end;
    ye[i] = visitor.segments[static_cast<std::size_t>(i)].y_end;
    depths[i] = visitor.segments[static_cast<std::size_t>(i)].depth;
  }

  return Rcpp::DataFrame::create(
    Rcpp::Named("x_start") = xs,
    Rcpp::Named("y_start") = ys,
    Rcpp::Named("x_end")   = xe,
    Rcpp::Named("y_end")   = ye,
    Rcpp::Named("depth")   = depths
  );
}

// [[Rcpp::export]]
Rcpp::List ppforest2_decision_regions(
  const Tree&             tree,
  Rcpp::IntegerVector     var_indices,
  Rcpp::NumericVector     fixed_values,
  double                  x_min,
  double                  x_max,
  double                  y_min,
  double                  y_max) {
  int vi = var_indices[0];
  int vj = var_indices[1];

  int p = static_cast<int>(var_indices.size()) + static_cast<int>(fixed_values.size());

  std::vector<std::pair<int, types::Feature>> fixed_vars;

  if (fixed_values.size() > 0) {
    std::set<int> used(var_indices.begin(), var_indices.end());
    int fv_idx = 0;

    for (int k = 0; k < p; ++k) {
      if (used.find(k) == used.end()) {
        fixed_vars.push_back({ k, static_cast<types::Feature>(fixed_values[fv_idx++]) });
      }
    }
  }

  RegionVisitor visitor(
    vi, vj, fixed_vars,
    static_cast<types::Feature>(x_min),
    static_cast<types::Feature>(x_max),
    static_cast<types::Feature>(y_min),
    static_cast<types::Feature>(y_max));

  tree.root->accept(visitor);

  Rcpp::List result(visitor.regions.size());

  for (std::size_t i = 0; i < visitor.regions.size(); ++i) {
    const auto& region = visitor.regions[i];

    Rcpp::NumericVector rx(region.vertices.size());
    Rcpp::NumericVector ry(region.vertices.size());

    for (std::size_t j = 0; j < region.vertices.size(); ++j) {
      rx[j] = region.vertices[j].first;
      ry[j] = region.vertices[j].second;
    }

    result[i] = Rcpp::List::create(
      Rcpp::Named("x")     = rx,
      Rcpp::Named("y")     = ry,
      Rcpp::Named("class") = region.predicted_class + 1  // C++ 0-based → R 1-based
    );
  }

  return result;
}

// [[Rcpp::export]]
Rcpp::List ppforest2_tree_layout(const Tree& tree) {
  LayoutParams params;
  TreeLayout layout = compute_tree_layout(*tree.root, params);

  // Build node data frame
  int n_nodes = static_cast<int>(layout.nodes.size());
  Rcpp::NumericVector nx(n_nodes), ny(n_nodes);
  Rcpp::LogicalVector n_leaf(n_nodes);
  Rcpp::IntegerVector n_idx(n_nodes);

  for (int i = 0; i < n_nodes; ++i) {
    nx[i] = layout.nodes[static_cast<std::size_t>(i)].x;
    ny[i] = layout.nodes[static_cast<std::size_t>(i)].y;
    n_leaf[i] = layout.nodes[static_cast<std::size_t>(i)].is_leaf;
    n_idx[i] = layout.nodes[static_cast<std::size_t>(i)].node_idx;
  }

  Rcpp::DataFrame node_df = Rcpp::DataFrame::create(
    Rcpp::Named("x")        = nx,
    Rcpp::Named("y")        = ny,
    Rcpp::Named("is_leaf")  = n_leaf,
    Rcpp::Named("node_idx") = n_idx
  );

  // Build edge data frame
  int n_edges = static_cast<int>(layout.edges.size());
  Rcpp::NumericVector efx(n_edges), efy(n_edges), etx(n_edges), ety(n_edges);
  Rcpp::CharacterVector elabel(n_edges);

  for (int i = 0; i < n_edges; ++i) {
    efx[i] = layout.edges[static_cast<std::size_t>(i)].from_x;
    efy[i] = layout.edges[static_cast<std::size_t>(i)].from_y;
    etx[i] = layout.edges[static_cast<std::size_t>(i)].to_x;
    ety[i] = layout.edges[static_cast<std::size_t>(i)].to_y;
    elabel[i] = layout.edges[static_cast<std::size_t>(i)].label;
  }

  Rcpp::DataFrame edge_df = Rcpp::DataFrame::create(
    Rcpp::Named("from_x")     = efx,
    Rcpp::Named("from_y")     = efy,
    Rcpp::Named("to_x")       = etx,
    Rcpp::Named("to_y")       = ety,
    Rcpp::Named("edge_label") = elabel
  );

  return Rcpp::List::create(
    Rcpp::Named("nodes") = node_df,
    Rcpp::Named("edges") = edge_df
  );
}

namespace {
  using json = nlohmann::json;

  json meta_from_r(
    Rcpp::CharacterVector classes,
    Rcpp::List            training_spec,
    int                   seed) {
    json meta;

    std::vector<std::string> cls(classes.begin(), classes.end());
    meta["classes"] = cls;

    std::string strategy = Rcpp::as<std::string>(training_spec["strategy"]);
    meta["training_spec"] = strategy;
    meta["lambda"]        = Rcpp::as<float>(training_spec["lambda"]);
    meta["seed"]          = seed;

    if (training_spec.containsElementNamed("n_vars")) {
      meta["n_vars"] = Rcpp::as<int>(training_spec["n_vars"]);
    }

    return meta;
  }

  FeatureVector to_feature_vector(Rcpp::NumericVector v) {
    int n = static_cast<int>(v.size());
    FeatureVector result(n);

    for (int i = 0; i < n; ++i) {
      result(i) = static_cast<Feature>(v[i]);
    }

    return result;
  }

  json vi_from_r(Rcpp::List vi) {
    VariableImportance cpp_vi;

    cpp_vi.scale       = to_feature_vector(vi["scale"]);
    cpp_vi.projections = to_feature_vector(vi["projections"]);

    if (vi.containsElementNamed("weighted")) {
      cpp_vi.weighted_projections = to_feature_vector(vi["weighted"]);
    }

    if (vi.containsElementNamed("permuted")) {
      cpp_vi.permuted = to_feature_vector(vi["permuted"]);
    }

    return ppforest2::serialization::to_json(cpp_vi);
  }
}

// [[Rcpp::export]]
std::string ppforest2_save_tree_json(
    const Tree&           tree,
    Rcpp::CharacterVector classes,
    Rcpp::List            vi,
    Rcpp::List            training_spec,
    int                   seed,
    bool                  include_metrics) {
  json output = ppforest2::serialization::to_json(tree);

  json result;
  result["model_type"] = "tree";
  result["meta"]       = meta_from_r(classes, training_spec, seed);
  result["model"]      = output;

  if (include_metrics) {
    result["variable_importance"] = vi_from_r(vi);
  }

  return result.dump(2);
}

// [[Rcpp::export]]
std::string ppforest2_save_forest_json(
    const Forest&         forest,
    Rcpp::CharacterVector classes,
    Rcpp::List            vi,
    Rcpp::List            training_spec,
    int                   seed,
    double                oob_error,
    bool                  include_metrics) {
  json output = ppforest2::serialization::to_json(forest);

  json result;
  result["model_type"] = "forest";
  result["meta"]       = meta_from_r(classes, training_spec, seed);
  result["model"]      = output;

  if (include_metrics) {
    result["variable_importance"] = vi_from_r(vi);
    result["oob_error"]           = oob_error;
  }

  return result.dump(2);
}

// [[Rcpp::export]]
Rcpp::List ppforest2_load_json_meta(const std::string& json_str) {
  json j = json::parse(json_str);

  // Detect model type: explicit field, meta.trees > 0, or model.trees key
  std::string model_type;

  if (j.contains("model_type")) {
    model_type = j["model_type"].get<std::string>();
  } else if (j.contains("model") && j["model"].contains("trees")) {
    model_type = "forest";
  } else if (j.contains("meta") && j["meta"].value("trees", 0) > 0) {
    model_type = "forest";
  } else {
    model_type = "tree";
  }

  // Restore class names from meta
  Rcpp::CharacterVector classes;
  int seed = 0;
  Rcpp::List training_spec;

  if (j.contains("meta")) {
    const auto& meta = j["meta"];

    if (meta.contains("classes")) {
      auto cls = meta["classes"].get<std::vector<std::string>>();
      classes = Rcpp::wrap(cls);
    }

    seed = meta.value("seed", 0);

    std::string strategy = meta.value("training_spec", "glda");
    float lambda         = meta.value("lambda", 0.0f);

    if (meta.contains("n_vars")) {
      training_spec = Rcpp::List::create(
        Rcpp::Named("strategy") = strategy,
        Rcpp::Named("lambda")   = lambda,
        Rcpp::Named("n_vars")   = meta["n_vars"].get<int>());
    } else {
      training_spec = Rcpp::List::create(
        Rcpp::Named("strategy") = strategy,
        Rcpp::Named("lambda")   = lambda);
    }
  }

  // Restore variable importance
  SEXP vi_sexp = R_NilValue;

  if (j.contains("variable_importance")) {
    const auto& vi_j = j["variable_importance"];

    Rcpp::List vi_list;
    if (vi_j.contains("scale")) {
      vi_list["scale"] = Rcpp::wrap(vi_j["scale"].get<std::vector<double>>());
    }

    if (vi_j.contains("projections")) {
      vi_list["projections"] = Rcpp::wrap(vi_j["projections"].get<std::vector<double>>());
    }

    if (vi_j.contains("weighted_projections")) {
      vi_list["weighted"] = Rcpp::wrap(vi_j["weighted_projections"].get<std::vector<double>>());
    }

    if (vi_j.contains("permuted")) {
      vi_list["permuted"] = Rcpp::wrap(vi_j["permuted"].get<std::vector<double>>());
    }

    vi_sexp = vi_list;
  }

  // OOB error (forest only)
  double oob_error = NA_REAL;
  if (j.contains("oob_error")) {
    oob_error = j["oob_error"].get<double>();
  }

  return Rcpp::List::create(
    Rcpp::Named("model_type")     = model_type,
    Rcpp::Named("classes")        = classes,
    Rcpp::Named("training_spec")  = training_spec,
    Rcpp::Named("seed")           = seed,
    Rcpp::Named("vi")             = vi_sexp,
    Rcpp::Named("oob_error")      = oob_error);
}

namespace {
  TrainingSpec::Ptr training_spec_from_meta(const json& j) {
    if (!j.contains("meta")) {
      return TrainingSpecGLDA::make(0.0f);
    }

    const auto& meta    = j["meta"];
    std::string strategy = meta.value("training_spec", "glda");
    float lambda         = meta.value("lambda", 0.0f);

    if (strategy == "uglda" || strategy == "uniform_glda") {
      int n_vars = meta.value("n_vars", 0);
      return TrainingSpecUGLDA::make(n_vars, lambda);
    }

    return TrainingSpecGLDA::make(lambda);
  }
}

// [[Rcpp::export]]
Tree ppforest2_tree_from_json(const std::string& json_str) {
  json j = json::parse(json_str);
  Tree tree = ppforest2::serialization::tree_from_json(j["model"]);
  tree.training_spec = training_spec_from_meta(j);
  return tree;
}

// [[Rcpp::export]]
Forest ppforest2_forest_from_json(const std::string& json_str) {
  json j = json::parse(json_str);
  Forest forest = ppforest2::serialization::forest_from_json(j["model"]);
  forest.training_spec = training_spec_from_meta(j);

  // Propagate training_spec to each tree
  for (auto& tree : forest.trees) {
    tree->training_spec = training_spec_from_meta(j);
  }

  return forest;
}
