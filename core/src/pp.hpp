#include "Data.hpp"
#include "DataColumn.hpp"
#include "DVector.hpp"

namespace pp {
  template<typename T>
  using Projector = DVector<T>;

  template<typename T>
  using Projection = DataColumn<T>;

  template<typename T, typename G>
  Projector<T> glda_optimum_projector(
    const Data<T> &      data,
    const DataColumn<G> &groups,
    const std::set<G> &  unique_groups,
    const double         lambda);


  template<typename T, typename G>
  T glda_index(
    const Data<T> &      data,
    const Projector<T> & projector,
    const DataColumn<G> &groups,
    const std::set<G> &  unique_groups,
    const double         lambda);

  template<typename T>
  Projection<T> project(
    const Data<T> &      data,
    const Projector<T> & projector);

  template<typename T>
  T project(
    const DataColumn<T> &data,
    const Projector<T> & projector);

  template<typename T>
  Projector<T> as_projector(std::vector<T> vector) {
    Eigen::Map<Projector<T> > projector(vector.data(), vector.size());
    return projector;
  }
}



namespace pp::strategy {
  template<typename T>
  using PPStrategyReturn = std::tuple<Projector<T>, Projection<T> >;
  template<typename T, typename G>
  using PPStrategy = std::function<PPStrategyReturn<T>(const Data<T>&, const DataColumn<G>&, const std::set<G>&)>;


  template<typename T, typename G>
  PPStrategy<T, G> glda(
    const double lambda);
}
