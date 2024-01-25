#include "stats.hpp"

namespace pp {
  template<typename T>
  using Projector = linalg::DVector<T>;

  template<typename T>
  using Projection = stats::DataColumn<T>;

  template<typename T>
  using PPStrategyReturn = std::tuple<Projector<T>, Projection<T> >;
  template<typename T, typename G>
  using PPStrategy = std::function<PPStrategyReturn<T>(stats::Data<T>, stats::DataColumn<G>, std::set<G>)>;

  template<typename T, typename G>
  Projector<T> lda_optimum_projector(
    stats::Data<T>       data,
    stats::DataColumn<G> groups,
    std::set<G>          unique_groups);

  template<typename T, typename G>
  T lda_index(
    stats::Data<T>       data,
    Projector<T>         projector,
    stats::DataColumn<G> groups,
    std::set<G>          unique_groups);

  template<typename T>
  Projection<T> project(
    stats::Data<T> data,
    Projector<T>   projector);

  template<typename T, typename G>
  PPStrategyReturn<T> lda_strategy(
    stats::Data<T>       data,
    stats::DataColumn<G> groups,
    std::set<G>          unique_groups);
}
