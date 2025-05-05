#include "Forest.hpp"
#include "Invariant.hpp"

#include <algorithm>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace models::stats;

namespace models {
  template<typename T, typename R >
  Forest<T, R> Forest<T, R>::train(
    const TrainingSpec<T, R> & training_spec,
    const Data<T> &            x,
    const DataColumn<R> &      y,
    const int                  size,
    const int                  seed,
    const int                  n_threads) {
    #ifdef _OPENMP
    omp_set_num_threads(n_threads);
    #endif

    Random::seed(seed);

    GroupSpec<T, R> training_data(x, y);

    LOG_INFO << "Training a random forest of " << size << " Project-Pursuit Trees." << std::endl;
    LOG_INFO << "The seed is: " << seed << std::endl;

    invariant(size > 0, "The forest size must be greater than 0.");

    Forest<T, R> forest(
      training_spec.clone(),
      x,
      y,
      training_data.groups,
      seed,
      n_threads);

    std::vector<BootstrapTreePtr<T, R> > trees(size);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
      std::vector<int> sample_indices = stratified_proportional_sample(
        x,
        y,
        training_data.groups,
        x.rows());

      trees[i] = BootstrapTree<T, R>::train(training_spec, x, y, sample_indices);
    }

    for (auto& tree : trees) {
      forest.add_tree(std::move(tree));
    }

    LOG_INFO << "Forest: " << forest << std::endl;

    return forest;
  }

  template Forest<float, int> Forest<float, int>::train(
    const TrainingSpec<float, int> & training_spec,
    const Data<float> &              x,
    const DataColumn<int> &          y,
    const int                        size,
    const int                        seed,
    const int                        n_threads);
}
