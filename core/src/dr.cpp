#include "dr.hpp"
#include "pptreeio.hpp"
#include <cassert>
#include <vector>

namespace dr::strategy {
  template<typename T>
  DRStrategy<T> all() {
    return [](const Data<T> &data, const std::mt19937 & rng) -> Data<T> {
             return data;
    };
  }

  template DRStrategy<long double> all<long double>();

  template<typename T>
  DRStrategy<T> uniform(int n_vars) {
    return [n_vars](const Data<T> &data, std::mt19937 & rng) -> Data<T> {
             assert(n_vars > 0 && "The number of variables must be greater than 0.");
             assert(n_vars <= data.cols() && "The number of variables must be less than or equal to the number of columns in the data.");

             if (n_vars == data.cols()) return data;

             LOG_INFO << "Selecting " << n_vars << " variables uniformly." << std::endl;

             std::vector<int> var_sampled_indices = Uniform(0, data.cols() - 1)(rng, n_vars);

             LOG_INFO << "Selected variables: " << var_sampled_indices << std::endl;

             Data<T> reduced_data = Data<T>::Zero(data.rows(), data.cols());

             for (int i = 0; i < n_vars; i++) {
               reduced_data.col(var_sampled_indices[i]) = data.col(var_sampled_indices[i]);
             }

             return reduced_data;
    };
  }

  template DRStrategy<long double> uniform<long double>(int);
}
