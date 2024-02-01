#include "pptree.hpp"

#include <RcppCommon.h>

namespace Rcpp {
  template<> SEXP wrap(const pptree::Node<double, int>& node);
  template<> SEXP wrap(const pptree::Tree<double, int>& tree);
}

namespace Rcpp::traits {
  template<> class Exporter<pptree::Node<double, int> >;
  template<> class Exporter<pptree::Tree<double, int> >;
}

#include <Rcpp.h>

namespace Rcpp {
  SEXP wrap(pptree::Node<double, int> node) {
    Rcpp::List rnode = Rcpp::List::create(
      Rcpp::Named("projector") = Rcpp::wrap(node.projector),
      Rcpp::Named("threshold") = Rcpp::wrap(node.threshold),
      Rcpp::Named("left") = Rcpp::wrap(*(node.left)),
      Rcpp::Named("right") = Rcpp::wrap(*(node.right)),
      Rcpp::Named("response") = node.response);

    return Rcpp::wrap(node);
  }

  SEXP wrap(pptree::Tree<double, int> tree) {
    Rcpp::List rtree = Rcpp::List::create(
      Rcpp::Named("root") = Rcpp::wrap(tree.root));

    return Rcpp::wrap(rtree);
  }
}

namespace Rcpp::traits {
  template<> class Exporter<pptree::Node<double, int> > {
    Rcpp::List rnode;

    public:
      Exporter(SEXP x) : rnode(x) {
      }

      pptree::Node<double, int> get() {
        pptree::Node<double, int> node;
        node.projector = Rcpp::as<pptree::pp::Projector<double> >(rnode["projector"]);
        node.threshold = Rcpp::as<pptree::Threshold<double> >(rnode["threshold"]);

        auto left_exporter = Exporter<pptree::Node<double, int> >(rnode["left"]);
        auto left_node = left_exporter.get();
        node.left = &left_node;

        auto right_exporter = Exporter<pptree::Node<double, int> >(rnode["right"]);
        auto right_node = right_exporter.get();
        node.right = &right_node;

        node.response = Rcpp::as<int>(rnode["response"]);
        return node;
      };
  };

  template<> class Exporter<pptree::Tree<double, int> > {
    Rcpp::List rtree;

    public:
      Exporter(SEXP x) : rtree(x) {
      }

      pptree::Tree<double, int> get() {
        pptree::Tree<double, int> tree;
        auto root_exporter = Exporter<pptree::Node<double, int> >(rtree["root"]);
        tree.root = root_exporter.get();
        return tree;
      }
  };
}
