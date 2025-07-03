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

    if (!GroupSpec<R>::is_contiguous(y)) {
      stats::sort(x, y);
    }

    GroupSpec<R> group_spec(y);

    invariant(size > 0, "The forest size must be greater than 0.");

    std::vector<BootstrapTreePtr<T, R> > trees(size);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
      stats::RNG rng(static_cast<uint64_t>(seed), static_cast<uint64_t>(i));
      trees[i] = BootstrapTree<T, R>::train(training_spec, x, group_spec, rng);
    }

    Forest<T, R> forest(training_spec.clone(), seed);

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
