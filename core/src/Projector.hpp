#pragma once

#include "Data.hpp"
#include "DataColumn.hpp"
#include "DVector.hpp"


namespace models::pp {
  template<typename T>
  using Projection = stats::DataColumn<T>;

  template<typename T>
  struct Projector {
    using Size = Eigen::Index;

    const math::DVector<T> vector;

    explicit Projector(math::DVector<T> vector) : vector(vector) {
    }

    Projector(std::initializer_list<T> init_list)
      : vector(math::DVector<T>::Map(init_list.begin(), init_list.size())) {
    }

    Projection<T> project(const stats::Data<T> & data) const {
      return data * vector;
    }

    T project(const stats::DataColumn<T> &data) const {
      return (data.transpose() * vector).value();
    }

    Projector<T> normalize()const {
      // Truncate the vector to avoid numerical instability
      math::DVector<T> truncated = vector.unaryExpr(reinterpret_cast<T (*)(T)>(&math::truncate<T>));

      // Fetch the index of the first non-zero component
      int i = 0;

      while (i < truncated.size() && math::is_approx(truncated(i), 0))
        i++;

      // Guarantee the first non-zero component is positive
      return Projector<T>((truncated(i) < 0 ? -1 : 1) * truncated);
    }

    Size size() const {
      return vector.size();
    }

    T operator()(Size i) const {
      return vector(i);
    }

    bool is_collinear(const Projector<T> &other) const {
      return math::collinear(vector, other.vector);
    }

    bool operator==(const Projector<T> &other) const {
      return vector.isApprox(other.vector);
    }

    Projector<T> expand(const std::vector<int> &mask) const {
      return Projector<T>(stats::expand(vector, mask));
    }

    json to_json() const {
      return vector;
    }
  };

  template<typename T>
  std::ostream& operator<<(std::ostream &ostream, const Projector<T> &projector) {
    return ostream << projector.vector;
  }
}
