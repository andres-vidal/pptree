#include "stats.hpp"

namespace pp {
  template<typename T>
  using Projector = linalg::DVector<T>;

  template<typename T>
  using Projection = stats::DataColumn<T>;

  template<typename T>
  using PPStrategyReturn = std::tuple<Projector<T>, Projection<T> >;
  template<typename T, typename G>
  using PPStrategy = std::function<PPStrategyReturn<T>(const stats::Data<T>&, const stats::DataColumn<G>&, const std::set<G>&)>;

  template<typename T, typename G>
  Projector<T> lda_optimum_projector(
    const stats::Data<T> &      data,
    const stats::DataColumn<G> &groups,
    const std::set<G> &         unique_groups);

  template<typename T, typename G>
  T lda_index(
    const stats::Data<T> &      data,
    const Projector<T> &        projector,
    const stats::DataColumn<G> &groups,
    const std::set<G> &         unique_groups);

  template<typename T>
  Projection<T> project(
    const stats::Data<T> &data,
    const Projector<T> &  projector);

  template<typename T>
  T project(
    const stats::DataColumn<T> &data,
    const Projector<T> &        projector);

  template<typename T, typename G>
  PPStrategyReturn<T> lda_strategy(
    const stats::Data<T> &      data,
    const stats::DataColumn<G> &groups,
    const std::set<G> &         unique_groups);

  template<typename T>
  Projector<T> as_projector(std::vector<T> vector) {
    Eigen::Map<Projector<T> > projector(vector.data(), vector.size());
    return projector;
  }
}
