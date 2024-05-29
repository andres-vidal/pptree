#include "pptreeio.hpp"
#include "pp.hpp"

#include "Math.hpp"


using namespace pp;

namespace pp {
  template<typename T>
  T truncate_op(T value) {
    return fabs(value) < 1e-15 ? 0 : value;
  }

  template<typename T>
  Projector<T> get_projector(Data<T> eigen_vec) {
    Projector<T> last = eigen_vec.col(eigen_vec.cols() - 1);
    Projector<T> truncated = last.unaryExpr(reinterpret_cast<T (*)(T)>(&truncate_op<T>));

    int i = 0;

    while (i < truncated.rows() && is_approx(truncated(i), 0))
      i++;

    return (truncated(i) < 0 ? -1 : 1) * truncated;
  }

  template<typename T, typename G>
  Projector<T> glda_optimum_projector(
    const Data<T> &      data,
    const DataColumn<G> &groups,
    const std::set<G> &  unique_groups,
    const double         lambda) {
    LOG_INFO << "Calculating PDA optimum projector for " << unique_groups.size() << " groups: " << unique_groups << std::endl;
    LOG_INFO << "Dataset size: " << data.rows() << " observations of " << data.cols() << " variables:" << std::endl;
    LOG_INFO << std::endl << data << std::endl;
    LOG_INFO << "Groups:" << std::endl;
    LOG_INFO << std::endl << groups << std::endl;

    Data<T> complete_B = between_groups_sum_of_squares(data, groups, unique_groups);
    Data<T> complete_W = within_groups_sum_of_squares(data, groups, unique_groups);

    LOG_INFO << "BGSS:" << std::endl << complete_B << std::endl;
    LOG_INFO << "WGSS:" << std::endl << complete_W << std::endl;

    auto [var_mask, var_index] = mask_null_columns(complete_B);

    LOG_INFO << "Considered variables after filtering out constant ones: " << var_index << std::endl;

    Data<T> B = complete_B(var_index, var_index);
    Data<T> W = complete_W(var_index, var_index);

    LOG_INFO << "B:" << std::endl << B << std::endl;
    LOG_INFO << "W:" << std::endl << W << std::endl;

    Data<T> W_diag = W.diagonal().asDiagonal();
    Data<T> W_pda = W_diag + (1 - lambda) * (W - W_diag);
    Data<T> WpB = W_pda + B;


    LOG_INFO << "W_pda:" << std::endl << W_pda << std::endl;
    LOG_INFO << "W_pda + B:" << std::endl << WpB << std::endl;

    Data<T> WpBInvB = solve(WpB, B);
    Data<T> truncatedWpBInvB = WpBInvB.unaryExpr(reinterpret_cast<T (*)(T)>(&truncate_op<T>));

    LOG_INFO << "(W_pda + B)^-1 * B:" << std::endl << WpBInvB << std::endl;
    LOG_INFO << "(W_pda + B)^-1 * B (truncated):" << std::endl << truncatedWpBInvB << std::endl;

    auto [eigen_val, eigen_vec] = eigen(truncatedWpBInvB);

    LOG_INFO << "Eigenvalues:" << std::endl << eigen_val << std::endl;
    LOG_INFO << "Eigenvectors:" << std::endl << eigen_vec << std::endl;

    Projector<T> projector = expand(get_projector(eigen_vec), var_mask);

    LOG_INFO << "Projector:" << std::endl << projector << std::endl;
    return projector;
  }

  template Projector<long double> glda_optimum_projector<long double, int>(
    const Data<long double> & data,
    const DataColumn<int> &   groups,
    const std::set<int> &     unique_groups,
    const double              lambda);


  template<typename T, typename G>
  T glda_index(
    const Data<T> &      data,
    const Projector<T> & projector,
    const DataColumn<G> &groups,
    const std::set<G> &  unique_groups,
    const double         lambda) {
    Data<T> A = projector;

    Data<T> W = within_groups_sum_of_squares(data, groups, unique_groups);
    Data<T> W_diag = W.diagonal().asDiagonal();
    Data<T> W_pda = W_diag + (1 - lambda) * (W - W_diag);
    Data<T> B = between_groups_sum_of_squares(data, groups, unique_groups);
    Data<T> WpB = W_pda + B;

    T denominator = determinant(inner_square(A, WpB));

    if (denominator == 0) {
      return 0;
    }

    return 1 - determinant(inner_square(A, W_pda)) / denominator;
  }

  template long double glda_index<long double, int>(
    const Data<long double> &     data,
    const Projector<long double> &projector,
    const DataColumn<int> &       groups,
    const std::set<int> &         unique_groups,
    const double                  lambda);

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
}

namespace pp::strategy {
  template<typename T, typename G>
  PPStrategy<T, G> glda(
    const double lambda) {
    if (lambda == 0) {
      LOG_INFO << "Chosen Projection-Pursuit Strategy is LDA" << std::endl;
    } else {
      LOG_INFO << "Chosen Projection-Pursuit Strategy is PDA(lambda = " << lambda << ")" << std::endl;
    }

    return [lambda](const Data<T>& data, const DataColumn<G>& groups, const std::set<G>& unique_groups) -> PPStrategyReturn<T> {
             auto projector = glda_optimum_projector(data, groups, unique_groups, lambda);
             return PPStrategyReturn<T> { projector, project(data, projector) };
    };
  }

  template PPStrategy<long double, int> glda<long double, int>(
    const double lambda);
}
