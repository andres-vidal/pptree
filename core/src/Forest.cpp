#include "Forest.hpp"
#include "Invariant.hpp"

#include <algorithm>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace models::stats;

namespace models {
  template<typename T, typename R>
  Forest<T, R> Forest<T, R>::train(
    const TrainingSpec<T, R> &training_spec,
    const DataSpec<T, R> &    training_data,
    const int                 size,
    const int                 seed) {
    return train(
      training_spec,
      training_data,
      size,
      seed,
      std::thread::hardware_concurrency());
  }

  template<typename T, typename R >
  Forest<T, R> Forest<T, R>::train(
    const TrainingSpec<T, R> &training_spec,
    const DataSpec<T, R> &    training_data,
    const int                 size,
    const int                 seed,
    const int                 n_threads) {
    #ifdef _OPENMP
    omp_set_num_threads(n_threads);
    #endif

    #pragma omp parallel
    {
      Random::seed(seed);
    }

    LOG_INFO << "Training a random forest of " << size << " Project-Pursuit Trees." << std::endl;
    LOG_INFO << "The seed is: " << seed << std::endl;

    invariant(size > 0, "The forest size must be greater than 0.");

    const SortedDataSpec<T, R> sorted_training_data(training_data);

    Forest<T, R> forest(
      training_spec.clone(),
      std::make_shared<SortedDataSpec<T, R> >(sorted_training_data),
      seed,
      n_threads);

    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
      auto sample = stratified_proportional_sample(sorted_training_data, sorted_training_data.x.rows());
      auto tree = BootstrapTree<T, R>::train(training_spec, sample);

      #pragma omp critical
      { forest.add_tree(std::make_unique<BootstrapTree<T, R> >(std::move(tree))); }
    }

    LOG_INFO << "Forest: " << forest << std::endl;

    return forest;
  }

  template Forest<double, int> Forest<double, int>::train(
    const TrainingSpec<double, int> &training_spec,
    const DataSpec<double, int> &    training_data,
    const int                        size,
    const int                        seed);

  template Forest<double, int> Forest<double, int>::train(
    const TrainingSpec<double, int> &training_spec,
    const DataSpec<double, int> &    training_data,
    const int                        size,
    const int                        seed,
    const int                        n_threads);
}
