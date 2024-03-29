#include "dr.hpp"
#include "pptreeio.hpp"
#include <cassert>
#include <vector>

using namespace stats;

namespace dr::strategy {
  template<typename T>
  DRStrategy<T> select_all_variables() {
    return [](const Data<T> &data) -> Data<T> {
             return data;
    };
  }

  template DRStrategy<long double> select_all_variables<long double>();

  template<typename T>
  DRStrategy<T> select_variables_uniformly(int n_vars, std::mt19937 &gen) {
    return [n_vars, &gen](const Data<T> &data) -> Data<T> {
             assert(n_vars > 0 && "The number of variables must be greater than 0.");
             assert(n_vars <= data.cols() && "The number of variables must be less than or equal to the number of columns in the data.");

             if (n_vars == data.cols()) return data;

             LOG_INFO << "Selecting " << n_vars << " variables uniformly." << std::endl;

             std::vector<int> var_sampled_indices = stats::Uniform(0, data.cols() - 1)(gen, n_vars);

             LOG_INFO << "Selected variables: " << var_sampled_indices << std::endl;

             Data<T> reduced_data = Data<T>::Zero(data.rows(), data.cols());

             for (int i = 0; i < n_vars; i++) {
               reduced_data.col(var_sampled_indices[i]) = data.col(var_sampled_indices[i]);
             }

             return reduced_data;
    };
  }

  template DRStrategy<long double> select_variables_uniformly<long double>(int, std::mt19937&);
}
