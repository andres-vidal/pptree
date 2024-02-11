#include "pptreeio.hpp"
#include "pp.hpp"


using namespace pp;
using namespace stats;
using namespace linalg;

namespace pp {
  template<typename T, typename G>
  Projector<T> lda_optimum_projector(
    Data<T>       data,
    DataColumn<G> groups,
    std::set<G>   unique_groups) {
    LOG_INFO << "Calculating LDA optimum projector for " << unique_groups.size() << " groups: " << unique_groups << std::endl;
    LOG_INFO << "Dataset size: " << data.rows() << " observations of " << data.cols() << " variables" << std::endl;

    Data<T> W = within_groups_sum_of_squares(data, groups, unique_groups);
    Data<T> B = between_groups_sum_of_squares(data, groups, unique_groups);

    LOG_INFO << "WGSS:" << std::endl << W << std::endl;
    LOG_INFO << "BGSS:" << std::endl << B << std::endl;

    auto [eigen_val, eigen_vec] = linalg::eigen(linalg::inverse(W + B) * B);

    Projector<T> projector = eigen_vec(Eigen::all, Eigen::last);

    LOG_INFO << "Projector:" << std::endl << projector << std::endl;
    return eigen_vec(Eigen::all, Eigen::last);
  }

  template Projector<double> lda_optimum_projector<double, int>(
    Data<double>    data,
    DataColumn<int> groups,
    std::set<int>   unique_groups);

  template<typename T, typename G>
  T lda_index(
    Data<T>       data,
    Projector<T>  projector,
    DataColumn<G> groups,
    std::set<G>   unique_groups) {
    Data<T> A = projector;

    Data<T> W = within_groups_sum_of_squares(data, groups, unique_groups);
    Data<T> B = between_groups_sum_of_squares(data, groups, unique_groups);

    T denominator = linalg::determinant(linalg::inner_square(A, W + B));

    if (denominator == 0) {
      return 0;
    }

    return 1 - determinant(inner_square(A, W)) / denominator;
  }

  template double lda_index<double, int>(
    Data<double>      data,
    Projector<double> projector,
    DataColumn<int>   groups,
    std::set<int>     unique_groups);

  template<typename T>
  Projection<T> project(
    Data<T>      data,
    Projector<T> projector) {
    return data * projector;
  }

  template Projection<double> project<double>(
    Data<double>      data,
    Projector<double> projector);

  template<typename T>
  T project(
    DataColumn<T> data,
    Projector<T>  projector) {
    return (data.transpose() * projector).value();
  }

  template double project<double>(
    DataColumn<double> data,
    Projector<double>  projector);



  template<typename T, typename G>
  PPStrategyReturn<T> lda_strategy(
    Data<T>       data,
    DataColumn<G> groups,
    std::set<G>   unique_groups) {
    Projector<T> projector = lda_optimum_projector(data, groups, unique_groups);
    return std::make_tuple(projector, project(data, projector));
  }

  template PPStrategyReturn<double> lda_strategy<double, int>(
    Data<double>    data,
    DataColumn<int> groups,
    std::set<int>   unique_groups);
}
