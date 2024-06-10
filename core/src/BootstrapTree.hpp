#include "Tree.hpp"
#include "BootstrapDataSpec.hpp"

namespace models {
  template<typename T, typename R>
  struct BootstrapTree : public Tree<T, R, stats::BootstrapDataSpec<T, R> > {
    explicit BootstrapTree(std::unique_ptr<Condition<T, R> > root)
      : Tree<T, R, stats::BootstrapDataSpec<T, R> >(std::move(root)) {
    }

    // cppcheck-suppress noExplicitConstructor
    BootstrapTree(Tree<T, R, stats::BootstrapDataSpec<T, R> > tree)
      : Tree<T, R, stats::BootstrapDataSpec<T, R> >(
        std::move(tree.root),
        std::move(tree.training_spec),
        std::move(tree.training_data)) {
    }

    BootstrapTree(
      std::unique_ptr<Condition<T, R> >                root,
      std::unique_ptr<TrainingSpec<T, R> >             training_spec,
      std::shared_ptr<stats::BootstrapDataSpec<T, R> > training_data)
      : Tree<T, R, stats::BootstrapDataSpec<T, R> >(std::move(root), std::move(training_spec), training_data) {
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
  };
}
