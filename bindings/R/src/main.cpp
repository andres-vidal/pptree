#include "../inst/include/ppforest2.h"
#include <RcppEigen.h>
#include "serialization/Json.hpp"

#include <nlohmann/json.hpp>
#include <fstream>
#include <thread>

// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
using namespace RcppEigen;
using namespace ppforest2;
using namespace ppforest2::types;
using namespace ppforest2::stats;
using namespace ppforest2::pp;
using namespace ppforest2::viz;
using namespace ppforest2::serialization;

// [[Rcpp::export]]
bool ppforest2_has_openmp() {
#ifdef _OPENMP
  return true;
#else
  return false;
#endif
}

// [[Rcpp::export]]
Model::Ptr ppforest2_train(TrainingSpec::Ptr spec, FeatureMatrix x, OutcomeVector y) {
  y.array() -= 1; // R 1-based → C++ 0-based

  if (!GroupPartition::is_contiguous(y)) {
    sort(x, y);
  }

  return Model::train(*spec, x, y);
}

// [[Rcpp::export]]
OutcomeVector ppforest2_predict_tree(Tree const& tree, FeatureMatrix const& data) {
  OutcomeVector result = tree.predict(data);
  result.array() += 1; // C++ 0-based → R 1-based
  return result;
}

// [[Rcpp::export]]
OutcomeVector ppforest2_predict_tree_forest(Forest const& forest, FeatureMatrix const& data) {
  OutcomeVector result = forest.predict(data);
  result.array() += 1; // C++ 0-based → R 1-based
  return result;
}

// [[Rcpp::export]]
FeatureMatrix ppforest2_predict_tree_prob(Tree const& tree, FeatureMatrix const& data) {
  return tree.predict(data, Proportions{});
}

// [[Rcpp::export]]
FeatureMatrix ppforest2_predict_forest_prob(Forest const& forest, FeatureMatrix const& data) {
  return forest.predict(data, Proportions{});
}

// [[Rcpp::export]]
FeatureVector ppforest2_vi_projections_tree(Tree const& tree, int n_vars, FeatureVector const& scale) {
  return variable_importance_projections(tree, n_vars, &scale);
}

// [[Rcpp::export]]
FeatureVector ppforest2_vi_projections_forest(Forest const& forest, int n_vars, FeatureVector const& scale) {
  return variable_importance_projections(forest, n_vars, &scale);
}

// [[Rcpp::export]]
FeatureVector ppforest2_vi_weighted_forest(
    Forest const& forest, FeatureMatrix const& x, OutcomeVector y, FeatureVector const& scale
) {
  y.array() -= 1; // R 1-based → C++ 0-based
  return variable_importance_weighted_projections(forest, x, y, &scale);
}

// [[Rcpp::export]]
FeatureVector ppforest2_vi_permuted_forest(Forest const& forest, FeatureMatrix const& x, OutcomeVector y, int seed) {
  y.array() -= 1; // R 1-based → C++ 0-based
  return variable_importance_permuted(forest, x, y, seed);
}

// [[Rcpp::export]]
double ppforest2_oob_error(Forest const& forest, FeatureMatrix const& x, OutcomeVector y) {
  y.array() -= 1; // R 1-based → C++ 0-based
  return forest.oob_error(x, y);
}

// [[Rcpp::export]]
OutcomeVector ppforest2_oob_predict(Forest const& forest, FeatureMatrix const& x) {
  OutcomeVector result = forest.oob_predict(x);
  // Convert C++ 0-based → R 1-based; sentinel -1 stays as 0 (handled in R)
  result.array() += 1;
  return result;
}

// [[Rcpp::export]]
Rcpp::List ppforest2_tree_node_data(Tree const& tree, FeatureMatrix const& x, OutcomeVector y) {
  y.array() -= 1; // R 1-based → C++ 0-based
  NodeDataVisitor visitor(x, y);
  tree.root->accept(visitor);

  Rcpp::List result(visitor.nodes.size());

  for (std::size_t i = 0; i < visitor.nodes.size(); ++i) {
    auto const& nd = visitor.nodes[i];

    // Convert 0-based group indices → R 1-based
    Rcpp::IntegerVector groups_r(nd.groups.begin(), nd.groups.end());
    groups_r = groups_r + 1;

    if (nd.is_leaf) {
      result[i] = Rcpp::List::create(
          Rcpp::Named("is_leaf") = true,
          Rcpp::Named("depth")   = nd.depth,
          Rcpp::Named("value")   = nd.value + 1,
          Rcpp::Named("groups")  = groups_r
      );
    } else {
      result[i] = Rcpp::List::create(
          Rcpp::Named("is_leaf")   = false,
          Rcpp::Named("depth")     = nd.depth,
          Rcpp::Named("projector") = Rcpp::wrap(nd.projector),
          Rcpp::Named("cutpoint")  = nd.cutpoint,
          Rcpp::Named("projected") = Rcpp::NumericVector(nd.projected_values.begin(), nd.projected_values.end()),
          Rcpp::Named("groups")    = groups_r
      );
    }
  }

  return result;
}

// [[Rcpp::export]]
Rcpp::DataFrame ppforest2_boundary_segments(
    Tree const& tree,
    Rcpp::IntegerVector var_indices,
    Rcpp::NumericVector fixed_values,
    double x_min,
    double x_max,
    double y_min,
    double y_max
) {
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
        fixed_vars.push_back({k, static_cast<types::Feature>(fixed_values[fv_idx++])});
      }
    }
  }

  BoundaryVisitor visitor(
      vi,
      vj,
      fixed_vars,
      static_cast<types::Feature>(x_min),
      static_cast<types::Feature>(x_max),
      static_cast<types::Feature>(y_min),
      static_cast<types::Feature>(y_max)
  );

  tree.root->accept(visitor);

  int n = static_cast<int>(visitor.segments.size());
  Rcpp::NumericVector xs(n), ys(n), xe(n), ye(n);
  Rcpp::IntegerVector depths(n);

  for (int i = 0; i < n; ++i) {
    xs[i]     = visitor.segments[static_cast<std::size_t>(i)].x_start;
    ys[i]     = visitor.segments[static_cast<std::size_t>(i)].y_start;
    xe[i]     = visitor.segments[static_cast<std::size_t>(i)].x_end;
    ye[i]     = visitor.segments[static_cast<std::size_t>(i)].y_end;
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
    Tree const& tree,
    Rcpp::IntegerVector var_indices,
    Rcpp::NumericVector fixed_values,
    double x_min,
    double x_max,
    double y_min,
    double y_max
) {
  int vi = var_indices[0];
  int vj = var_indices[1];

  int p = static_cast<int>(var_indices.size()) + static_cast<int>(fixed_values.size());

  std::vector<std::pair<int, types::Feature>> fixed_vars;

  if (fixed_values.size() > 0) {
    std::set<int> used(var_indices.begin(), var_indices.end());
    int fv_idx = 0;

    for (int k = 0; k < p; ++k) {
      if (used.find(k) == used.end()) {
        fixed_vars.push_back({k, static_cast<types::Feature>(fixed_values[fv_idx++])});
      }
    }
  }

  RegionVisitor visitor(
      vi,
      vj,
      fixed_vars,
      static_cast<types::Feature>(x_min),
      static_cast<types::Feature>(x_max),
      static_cast<types::Feature>(y_min),
      static_cast<types::Feature>(y_max)
  );

  tree.root->accept(visitor);

  Rcpp::List result(visitor.regions.size());

  for (std::size_t i = 0; i < visitor.regions.size(); ++i) {
    auto const& region = visitor.regions[i];

    Rcpp::NumericVector rx(region.vertices.size());
    Rcpp::NumericVector ry(region.vertices.size());

    for (std::size_t j = 0; j < region.vertices.size(); ++j) {
      rx[j] = region.vertices[j].first;
      ry[j] = region.vertices[j].second;
    }

    result[i] = Rcpp::List::create(
        Rcpp::Named("x")     = rx,
        Rcpp::Named("y")     = ry,
        Rcpp::Named("group") = region.predicted_group + 1 // C++ 0-based → R 1-based
    );
  }

  return result;
}

// [[Rcpp::export]]
Rcpp::List ppforest2_tree_layout(Tree const& tree) {
  LayoutParams params;
  TreeLayout layout = compute_tree_layout(*tree.root, params);

  // Build node data frame
  int n_nodes = static_cast<int>(layout.nodes.size());
  Rcpp::NumericVector nx(n_nodes), ny(n_nodes);
  Rcpp::LogicalVector n_leaf(n_nodes);
  Rcpp::IntegerVector n_idx(n_nodes);

  for (int i = 0; i < n_nodes; ++i) {
    nx[i]     = layout.nodes[static_cast<std::size_t>(i)].x;
    ny[i]     = layout.nodes[static_cast<std::size_t>(i)].y;
    n_leaf[i] = layout.nodes[static_cast<std::size_t>(i)].is_leaf;
    n_idx[i]  = layout.nodes[static_cast<std::size_t>(i)].node_idx;
  }

  Rcpp::DataFrame node_df = Rcpp::DataFrame::create(
      Rcpp::Named("x") = nx, Rcpp::Named("y") = ny, Rcpp::Named("is_leaf") = n_leaf, Rcpp::Named("node_idx") = n_idx
  );

  // Build edge data frame
  int n_edges = static_cast<int>(layout.edges.size());
  Rcpp::NumericVector efx(n_edges), efy(n_edges), etx(n_edges), ety(n_edges);
  Rcpp::CharacterVector elabel(n_edges);

  for (int i = 0; i < n_edges; ++i) {
    efx[i]    = layout.edges[static_cast<std::size_t>(i)].from_x;
    efy[i]    = layout.edges[static_cast<std::size_t>(i)].from_y;
    etx[i]    = layout.edges[static_cast<std::size_t>(i)].to_x;
    ety[i]    = layout.edges[static_cast<std::size_t>(i)].to_y;
    elabel[i] = layout.edges[static_cast<std::size_t>(i)].label;
  }

  Rcpp::DataFrame edge_df = Rcpp::DataFrame::create(
      Rcpp::Named("from_x")     = efx,
      Rcpp::Named("from_y")     = efy,
      Rcpp::Named("to_x")       = etx,
      Rcpp::Named("to_y")       = ety,
      Rcpp::Named("edge_label") = elabel
  );

  return Rcpp::List::create(Rcpp::Named("nodes") = node_df, Rcpp::Named("edges") = edge_df);
}

namespace {
  using json = nlohmann::json;
}

// [[Rcpp::export]]
std::string ppforest2_save_model_json(
    Model::Ptr model,
    std::vector<std::string> groups,
    bool include_metrics,
    FeatureMatrix const& x,
    OutcomeVector y,
    std::vector<std::string> feature_names
) {
  y.array() -= 1;

  Export<Model::Ptr> model_export{
      std::move(model),
      std::move(groups),
      nullptr,
      static_cast<int>(x.rows()),
      static_cast<int>(x.cols()),
      std::move(feature_names),
  };

  if (include_metrics) {
    model_export.compute_metrics(x, y);
  }

  return model_export.to_json().dump(2);
}

// [[Rcpp::export]]
ppforest2::serialization::Export<Model::Ptr> ppforest2_load_model_json(std::string const& path) {
  std::ifstream in(path);

  if (!in.is_open()) {
    Rcpp::stop("Could not open file: " + path);
  }

  auto j = json::parse(in);
  return j.get<Export<Model::Ptr>>();
}
