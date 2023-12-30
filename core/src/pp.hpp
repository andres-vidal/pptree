#include "stats.hpp"

using namespace linear_algebra;
using namespace stats;
namespace pp {
DVector<double> lda_optimum_projector(
  DMatrix<double>         data,
  DVector<unsigned short> groups,
  unsigned int            group_count);

double lda_index(
  DMatrix<double>         data,
  DMatrix<double>         projection_vector,
  DVector<unsigned short> groups,
  unsigned int            group_count);
}
