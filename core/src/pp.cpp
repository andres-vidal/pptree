#include "pptreeio.hpp"
#include "pp.hpp"


using namespace pp;
using namespace stats;
using namespace linalg;

namespace pp {
  template<typename T, typename G>
  Projector<T> lda_optimum_projector(
    const Data<T> &      data,
    const DataColumn<G> &groups,
    const std::set<G> &  unique_groups) {
    LOG_INFO << "Calculating LDA optimum projector for " << unique_groups.size() << " groups: " << unique_groups << std::endl;
    LOG_INFO << "Dataset size: " << data.rows() << " observations of " << data.cols() << " variables" << std::endl;

    Data<T> W = within_groups_sum_of_squares(data, groups, unique_groups);
    Data<T> B = between_groups_sum_of_squares(data, groups, unique_groups);

    LOG_INFO << "W:" << std::endl << W << std::endl;
    LOG_INFO << "B:" << std::endl << B << std::endl;

    Data<T> WpB = W + B;
    Data<T> WpBInv = linalg::inverse(WpB);
    Data<T> WpBInvB = WpBInv * B;

    LOG_INFO << "W + B:" << std::endl << WpB << std::endl;
    LOG_INFO << "(W + B)^-1" << std::endl << WpBInv << std::endl;
    LOG_INFO << "(W + B)^-1 * B:" << std::endl << WpBInvB << std::endl;

    auto [eigen_val, eigen_vec] = linalg::eigen(WpBInvB);

    LOG_INFO << "Eigenvalues:" << std::endl << eigen_val << std::endl;
    LOG_INFO << "Eigenvectors:" << std::endl << eigen_vec << std::endl;

    Projector<T> projector = eigen_vec(Eigen::all, Eigen::last);

    LOG_INFO << "Projector:" << std::endl << projector << std::endl;
    return eigen_vec(Eigen::all, Eigen::last);
  }

  template Projector<long double> lda_optimum_projector<long double, int>(
    const Data<long double> & data,
    const DataColumn<int> &   groups,
    const std::set<int> &     unique_groups);

  template<typename T, typename G>
  T lda_index(
    const Data<T> &      data,
    const Projector<T> & projector,
    const DataColumn<G> &groups,
    const std::set<G> &  unique_groups) {
    Data<T> A = projector;

    Data<T> W = within_groups_sum_of_squares(data, groups, unique_groups);
    Data<T> B = between_groups_sum_of_squares(data, groups, unique_groups);

    T denominator = linalg::determinant(linalg::inner_square(A, W + B));

    if (denominator == 0) {
      return 0;
    }

    return 1 - determinant(inner_square(A, W)) / denominator;
  }

  template long double lda_index<long double, int>(
    const Data<long double> &     data,
    const Projector<long double> &projector,
    const DataColumn<int> &       groups,
    const std::set<int> &         unique_groups);

  template<typename T>
  Projection<T> project(
    const Data<T> &     data,
    const Projector<T> &projector) {
    return data * projector;
  }

  template Projection<long double> project<long double>(
    const Data<long double> &     data,
    const Projector<long double> &projector);

  template<typename T>
  T project(
    const DataColumn<T> &data,
    const Projector<T> & projector) {
    return (data.transpose() * projector).value();
  }

  template long double project<long double>(
    const DataColumn<long double> &data,
    const Projector<long double> & projector);

  template<typename T, typename G>
  PPStrategyReturn<T> lda_strategy(
    const Data<T> &      data,
    const DataColumn<G> &groups,
    const std::set<G> &  unique_groups) {
    Projector<T> projector = lda_optimum_projector(data, groups, unique_groups);
    return std::make_tuple(projector, project(data, projector));
  }

  template PPStrategyReturn<long double> lda_strategy<long double, int>(
    const Data<long double> & data,
    const DataColumn<int> &   groups,
    const std::set<int> &     unique_groups);
}
