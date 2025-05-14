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
    Data<T> &                  x,
    DataColumn<R> &            y,
    const int                  size,
    const int                  seed,
    const int                  n_threads) {
    #ifdef _OPENMP
    omp_set_num_threads(n_threads);
    #endif

    Random::seed(seed);

    if (!GroupSpec<R>::is_contiguous(y)) {
      stats::sort(x, y);
    }

    GroupSpec<R> data_spec(y);

    invariant(size > 0, "The forest size must be greater than 0.");

    std::vector<BootstrapTreePtr<T, R> > trees(size);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
      std::vector<int> iob_indices;
      iob_indices.reserve(x.rows());

      for (const auto& group : data_spec.groups) {
        const int group_size = data_spec.group_size(group);
        const int min_index  = data_spec.group_start(group);
        const int max_index  = data_spec.group_end(group);

        const Uniform unif(min_index, max_index);

        for (int j = 0; j < group_size; j++) {
          iob_indices.push_back(unif());
        }
      }

      std::sort(iob_indices.begin(), iob_indices.end());

      trees[i] = BootstrapTree<T, R>::train(training_spec, x, y, iob_indices);
    }

    Forest<T, R> forest(
      training_spec.clone(),
      x,
      y,
      data_spec.groups,
      seed,
      n_threads);

    for (auto& tree : trees) {
      forest.add_tree(std::move(tree));
    }

    return forest;
  }

  template Forest<float, int> Forest<float, int>::train(
    const TrainingSpec<float, int> & training_spec,
    Data<float> &                    x,
    DataColumn<int> &                y,
    const int                        size,
    const int                        seed,
    const int                        n_threads);
}
