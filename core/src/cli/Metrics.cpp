/**
 * @file Metrics.cpp
 * @brief Compute model metrics and add them to a JSON model representation.
 */
#include "cli/Metrics.hpp"
#include "models/VariableImportance.hpp"
#include "stats/ConfusionMatrix.hpp"
#include "stats/Stats.hpp"
#include "serialization/Json.hpp"

using namespace ppforest2::types;
using namespace ppforest2::stats;

namespace ppforest2::cli {
namespace {
  struct MetricsVisitor : Model::Visitor {
    const FeatureMatrix& x;
    const ResponseVector& y;
    int n_vars;
    int seed;
    VariableImportance& vi;
    nlohmann::json& model_data;
    const std::vector<std::string>& group_names;

    MetricsVisitor(const FeatureMatrix& x, const ResponseVector& y,
    int n_vars, int seed, VariableImportance & vi,
    nlohmann::json& model_data,
    const std::vector<std::string>& group_names) :
      x(x), y(y), n_vars(n_vars), seed(seed), vi(vi),
      model_data(model_data), group_names(group_names) {
    }

    void visit(const Forest& forest) override {
      ResponseVector oob_preds = forest.oob_predict(x);

      std::vector<int> oob_rows;

      for (int i = 0; i < oob_preds.size(); ++i) {
        if (oob_preds(i) >= 0) {
          oob_rows.push_back(i);
        }
      }

      if (!oob_rows.empty()) {
        ResponseVector preds_oob = oob_preds(oob_rows, Eigen::all).eval();
        ResponseVector y_oob     = y(oob_rows, Eigen::all).eval();
        double oob_err           = stats::error_rate(preds_oob, y_oob);
        model_data["oob_error"] = oob_err;

        ConfusionMatrix oob_cm(preds_oob, y_oob);
        model_data["oob_confusion_matrix"] = group_names.empty()
          ? serialization::to_json(oob_cm)
          : serialization::to_json(oob_cm, group_names);
      }

      vi.permuted             = variable_importance_permuted(forest, x, y, seed);
      vi.projections          = variable_importance_projections(forest, n_vars, &vi.scale);
      vi.weighted_projections = variable_importance_weighted_projections(forest, x, y, &vi.scale);
    }

    void visit(const Tree& tree) override {
      vi.projections = variable_importance_projections(tree, n_vars, &vi.scale);
    }
  };
}

  void compute_metrics(
    nlohmann::json&                 model_data,
    const Model&                    model,
    const FeatureMatrix&            x,
    const ResponseVector&           y,
    const std::vector<std::string>& group_names,
    int                             seed) {
    const int n_vars = static_cast<int>(x.cols());

    VariableImportance vi;
    vi.scale = stats::sd(x);
    vi.scale = (vi.scale.array() > Feature(0)).select(vi.scale, Feature(1));

    // Training confusion matrix
    ResponseVector train_preds = model.predict(x);
    ConfusionMatrix train_cm(train_preds, y);

    model_data["training_confusion_matrix"] = group_names.empty()
      ? serialization::to_json(train_cm)
      : serialization::to_json(train_cm, group_names);

    MetricsVisitor metrics_visitor(x, y, n_vars, seed, vi, model_data, group_names);
    model.accept(metrics_visitor);

    model_data["variable_importance"] = serialization::to_json(vi);
  }
}
