#include "stats.hpp"
namespace pp {
template<typename T = double>
using Projector = linalg::DVector<T>;

template<typename T = double>
using Threshold = T;

template<typename T = double>
using Data = linalg::DMatrix<T>;

linalg::DVector<double> lda_optimum_projector(
  linalg::DMatrix<double>         data,
  linalg::DVector<unsigned short> groups,
  unsigned int                    group_count);

double lda_index(
  linalg::DMatrix<double>         data,
  linalg::DMatrix<double>         projection_vector,
  linalg::DVector<unsigned short> groups,
  unsigned int                    group_count);
}
