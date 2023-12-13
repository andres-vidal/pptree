#include "stats.hpp"
namespace pp {
template<typename T = double>
using Projector = linalg::DVector<T>;

template<typename T = double>
using Threshold = T;

template<typename T = double>
using Data = linalg::DMatrix<T>;

template<typename T = double>
using DataColumn = linalg::DVector<T>;

Projector<double> lda_optimum_projector(
  Data<double>               data,
  DataColumn<unsigned short> groups,
  unsigned int               group_count);

double lda_index(
  Data<double>               data,
  Projector<double>          projector,
  DataColumn<unsigned short> groups,
  unsigned int               group_count);

DataColumn<double> project(
  Data<double>      data,
  Projector<double> projector);
}
