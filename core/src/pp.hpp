#include "stats.hpp"
namespace pp {
template<typename T = double>
using Projector = linear_algebra::DVector<T>;

template<typename T = double>
using Threshold = T;

template<typename T = double>
using Data = linear_algebra::DMatrix<T>;

linear_algebra::DVector<double> lda_optimum_projector(
  linear_algebra::DMatrix<double>         data,
  linear_algebra::DVector<unsigned short> groups,
  unsigned int                            group_count);

double lda_index(
  linear_algebra::DMatrix<double>         data,
  linear_algebra::DMatrix<double>         projection_vector,
  linear_algebra::DVector<unsigned short> groups,
  unsigned int                            group_count);
}
