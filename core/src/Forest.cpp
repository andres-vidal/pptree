#include "Forest.hpp"

using namespace models::stats;

namespace models {
  template<typename T, typename R >
  Forest<T, R> Forest<T, R>::train(
    const TrainingSpec<T, R> &training_spec,
    const DataSpec<T, R> &    training_data,
    const int                 size,
    const int                 seed) {
    LOG_INFO << "Training a random forest of " << size << " Project-Pursuit Trees." << std::endl;
    LOG_INFO << "The seed is: " << seed << std::endl;

    assert(size > 0 && "The forest size must be greater than 0.");

    Random::rng.seed(seed);

    Forest<T, R> forest(
      training_spec.clone(),
      std::make_shared<DataSpec<T, R> >(training_data),
      seed);

    for (int i = 0; i < size; i++) {
      BootstrapDataSpec<T, R> sample_training_data = stratified_proportional_sample(
        training_data,
        training_data.x.rows());

      BootstrapTree<T, R> tree = BootstrapTree<T, R>::train(
        training_spec,
        sample_training_data);

      forest.add_tree(std::make_unique<BootstrapTree<T, R> >(std::move(tree)));
    }

    LOG_INFO << "Forest: " << forest << std::endl;

    return forest;
  }

  template Forest<long double, int> Forest<long double, int>::train(
    const TrainingSpec<long double, int> &training_spec,
    const DataSpec<long double, int> &    training_data,
    const int                             size,
    const int                             seed);
}
