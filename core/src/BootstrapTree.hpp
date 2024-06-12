#include "Tree.hpp"
#include "BootstrapDataSpec.hpp"

namespace models {
  template<typename T, typename R>
  struct BootstrapTree : public Tree<T, R, stats::BootstrapDataSpec<T, R> > {
    using Base = Tree<T, R, stats::BootstrapDataSpec<T, R> >;
    using Base::variable_importance;


    explicit BootstrapTree(std::unique_ptr<Condition<T, R> > root)
      : Tree<T, R, stats::BootstrapDataSpec<T, R> >(std::move(root)) {
    }

    // cppcheck-suppress noExplicitConstructor
    BootstrapTree(Tree<T, R, stats::BootstrapDataSpec<T, R> > tree)
      : Base(std::move(tree.root), std::move(tree.training_spec), std::move(tree.training_data)) {
    }

    BootstrapTree(
      std::unique_ptr<Condition<T, R> >                root,
      std::unique_ptr<TrainingSpec<T, R> >             training_spec,
      std::shared_ptr<stats::BootstrapDataSpec<T, R> > training_data)
      : Base(std::move(root),  std::move(training_spec), training_data) {
    }

    double error_rate(const stats::DataSpec<T, R> &data) const override {
      return Tree<T, R, stats::BootstrapDataSpec<T, R> >::error_rate(data);
    }

    double error_rate() const {
      return error_rate(this->training_data->get_oob());
    }

    stats::ConfusionMatrix confusion_matrix(const stats::DataSpec<T, R> &data) const override {
      return Tree<T, R, stats::BootstrapDataSpec<T, R> >::confusion_matrix(data);
    }

    stats::ConfusionMatrix confusion_matrix() const {
      return confusion_matrix(this->training_data->get_oob());
    }

    math::DVector<T> variable_importance(VariableImportanceKind importance_kind) const override {
      if (importance_kind == VariableImportanceKind::PERMUTATION) {
        return permutation_variable_importance();
      }

      math::DVector<T> importance = Base::variable_importance(importance_kind);

      double factor = 1.0;

      if (importance_kind == VariableImportanceKind::PROJECTOR_ADJUSTED) {
        factor = (1.0 - error_rate());
      }

      return importance * factor;
    }

    private:
      math::DVector<T> permutation_variable_importance() const {
        const stats::DataSpec<T, R> oob = this->training_data->get_oob();
        const stats::DataColumn<R> oob_predictions = Base::predict(oob.x);
        const double oob_accuracy = stats::accuracy(oob_predictions, oob.y);

        math::DVector<T> importance = math::DVector<T>(oob.x.cols());

        for (int j = 0; j < oob.x.cols(); j++) {
          const stats::Data<T> nonsense_data = stats::shuffle_column(oob.x, j);
          const stats::DataColumn<R> nonsense_predictions = Base::predict(nonsense_data);
          const double nonsense_accuracy = stats::accuracy(nonsense_predictions, oob.y);

          importance(j) = oob_accuracy - nonsense_accuracy;
        }

        return importance;
      }
  };
}
