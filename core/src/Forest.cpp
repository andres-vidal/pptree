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
    const TrainingSpec<T, R> &   training_spec,
    const SortedDataSpec<T, R> & training_data,
    const int                    size,
    const int                    seed) {
    return train(
      training_spec,
      training_data,
      size,
      seed,
      std::thread::hardware_concurrency());
  }

  template<typename T, typename R >
  Forest<T, R> Forest<T, R>::train(
    const TrainingSpec<T, R> &   training_spec,
    const SortedDataSpec<T, R> & training_data,
    const int                    size,
    const int                    seed,
    const int                    n_threads) {
    #ifdef _OPENMP
    omp_set_num_threads(n_threads);
    #endif

    Random::seed(seed);

    LOG_INFO << "Training a random forest of " << size << " Project-Pursuit Trees." << std::endl;
    LOG_INFO << "The seed is: " << seed << std::endl;

    invariant(size > 0, "The forest size must be greater than 0.");

    Forest<T, R> forest(
      training_spec.clone(),
      training_data.x,
      training_data.y,
      training_data.classes,
      seed,
      n_threads);

    std::vector<BootstrapTreePtr<T, R> > trees(size);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
      std::vector<int> sample_indices = stratified_proportional_sample(
        training_data.x,
        training_data.y,
        training_data.classes,
        training_data.x.rows());

      trees[i] = BootstrapTree<T, R>::train(training_spec, training_data, sample_indices);
    }

    for (auto& tree : trees) {
      forest.add_tree(std::move(tree));
    }

    LOG_INFO << "Forest: " << forest << std::endl;

    return forest;
  }

  template Forest<float, int> Forest<float, int>::train(
    const TrainingSpec<float, int> &   training_spec,
    const SortedDataSpec<float, int> & training_data,
    const int                          size,
    const int                          seed);

  template Forest<float, int> Forest<float, int>::train(
    const TrainingSpec<float, int> &   training_spec,
    const SortedDataSpec<float, int> & training_data,
    const int                          size,
    const int                          seed,
    const int                          n_threads);
}
