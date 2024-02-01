#include "pptree.hpp"
#include <iostream>

#include <RcppCommon.h>

namespace Rcpp {
  SEXP wrap(const pptree::Node<long double, int>& node);
  SEXP wrap(const pptree::Tree<long double, int> tree);
  SEXP wrap(const pptree::Response<long double, int> node);
  SEXP wrap(const pptree::Condition<long double, int> node);
}

namespace Rcpp::traits {
  template<> class Exporter<pptree::Response<long double, int> >;
  template<> class Exporter<pptree::Condition<long double, int> >;
  template<> class Exporter<pptree::Tree<long double, int> >;
}

#include <Rcpp.h>

namespace Rcpp {
  SEXP wrap(const pptree::Node<long double, int>& node) {
    if (node.is_response()) {
      return wrap(pptree::as_response(node));
    } else {
      return wrap(pptree::as_condition(node));
    }
  }

  SEXP wrap(const pptree::Response<long double, int> &node) {
    return Rcpp::List::create(
      Rcpp::Named("value") = Rcpp::wrap(node.value));
  }

  SEXP wrap(const pptree::Condition<long double, int> &node) {
    return Rcpp::List::create(
      Rcpp::Named("projector") = Rcpp::wrap(node.projector),
      Rcpp::Named("threshold") = Rcpp::wrap(node.threshold),
      Rcpp::Named("lower") = Rcpp::wrap(*node.lower),
      Rcpp::Named("upper") = Rcpp::wrap(*node.upper));
  }

  SEXP wrap(const pptree::Tree<long double, int> &tree) {
    return Rcpp::List::create(
      Rcpp::Named("root") = Rcpp::wrap(*tree.root));
  }
}

namespace Rcpp::traits {
  template<> class Exporter<pptree::Condition<long double, int> > {
    Rcpp::List rnode;

    public:
      Exporter(SEXP x) : rnode(x) {
      }

      pptree::Condition<long double, int> get() {
        auto projector = Rcpp::as<pptree::pp::Projector<long double> >(rnode["projector"]);
        auto threshold = Rcpp::as<pptree::Threshold<long double> >(rnode["threshold"]);

        auto lower_exporter = Exporter<pptree::Condition<long double, int> >(rnode["upper"]);
        auto lower_node = lower_exporter.get();

        auto upper_exporter = Exporter<pptree::Condition<long double, int> >(rnode["lower"]);
        auto upper_node = upper_exporter.get();

        return pptree::Condition<long double, int>(
          projector,
          threshold,
          &lower_node,
          &upper_node);
      };
  };

  template<> class Exporter<pptree::Response<long double, int> > {
    Rcpp::List rnode;

    public:
      Exporter(SEXP x) : rnode(x) {
      }

      pptree::Response<long double, int> get() {
        auto value = Rcpp::as<int>(rnode["value"]);
        return pptree::Response<long double, int>(value);
      }
  };

  template<> class Exporter<pptree::Tree<long double, int> > {
    Rcpp::List rtree;

    public:
      Exporter(SEXP x) : rtree(x) {
      }

      pptree::Tree<long double, int> get() {
        auto root_exporter = Exporter<pptree::Condition<long double, int> >(rtree["root"]);
        auto root_node = root_exporter.get();
        return pptree::Tree<long double, int>(&root_node);
      }
  };
}
