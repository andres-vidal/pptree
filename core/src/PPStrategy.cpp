#include "PPStrategy.hpp"

using namespace pptree::stats;
using namespace pptree::math;

namespace pptree::pp::strategy {
  template<typename T>
  Projector<T> get_projector(Data<T> eigen_vec) {
    Projector<T> last = eigen_vec.col(eigen_vec.cols() - 1);
    Projector<T> truncated = last.unaryExpr(reinterpret_cast<T (*)(T)>(&truncate<T>));

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
    Data<T> truncatedWpBInvB = WpBInvB.unaryExpr(reinterpret_cast<T (*)(T)>(&truncate<T>));

    LOG_INFO << "(W_pda + B)^-1 * B:" << std::endl << WpBInvB << std::endl;
    LOG_INFO << "(W_pda + B)^-1 * B (truncated):" << std::endl << truncatedWpBInvB << std::endl;

    auto [eigen_val, eigen_vec] = eigen(truncatedWpBInvB);

    LOG_INFO << "Eigenvalues:" << std::endl << eigen_val << std::endl;
    LOG_INFO << "Eigenvectors:" << std::endl << eigen_vec << std::endl;

    Projector<T> projector = expand(get_projector(eigen_vec), var_mask);

    LOG_INFO << "Projector:" << std::endl << projector << std::endl;
    return projector;
  }

  template Projector<long double> glda_optimum_projector(
    const Data<long double> & data,
    const DataColumn<int> &   groups,
    const std::set<int> &     unique_groups,
    const double              lambda);
}
