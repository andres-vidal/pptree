#include "pptree.hpp"

#include <RcppCommon.h>

using namespace pptree;

namespace Rcpp {
  SEXP wrap(const Node<long double, int> &);
  SEXP wrap(const Tree<long double, int > &);
  SEXP wrap(const BootstrapTree<long double, int> &);
  SEXP wrap(const Response<long double, int> &);
  SEXP wrap(const Condition<long double, int> &);
  SEXP wrap(const Forest<long double, int> &);

  SEXP wrap(const TrainingSpec<long double, int> &);
  SEXP wrap(const GLDATrainingSpec<long double, int> &);
  SEXP wrap(const UniformGLDATrainingSpec<long double, int> &);

  SEXP wrap(const DataSpec<long double, int> &);
  SEXP wrap(const BootstrapDataSpec<long double, int> &);

  template<> std::unique_ptr<Node<long double, int> > as(SEXP);
  template<> Tree<long double, int > as(SEXP);
  template<> BootstrapTree<long double, int> as(SEXP);
  template<> Response<long double, int> as(SEXP);
  template<> Condition<long double, int> as(SEXP);
  template<> Forest<long double, int> as(SEXP);

  template<> std::unique_ptr<TrainingSpec<long double, int> > as(SEXP);

  template<> DataSpec<long double, int>  as(SEXP);
  template<> BootstrapDataSpec<long double, int> as(SEXP);
}


#include <Rcpp.h>

namespace Rcpp {
  SEXP wrap(const Node<long double, int>& node) {
    struct NodeWrapper : public NodeVisitor<long double, int> {
      Rcpp::List result;

      void visit(const Condition<long double, int> &condition) {
        result = Rcpp::wrap(condition);
      }

      void visit(const Response<long double, int> &response) {
        result = Rcpp::wrap(response);
      }
    };

    NodeWrapper wrapper;
    node.accept(wrapper);
    return wrapper.result;
  }

  SEXP wrap(const Response<long double, int> &node) {
    return Rcpp::List::create(
      Rcpp::Named("value") = Rcpp::wrap(node.value));
  }

  SEXP wrap(const Condition<long double, int> &node) {
    return Rcpp::List::create(
      Rcpp::Named("projector") = Rcpp::wrap(node.projector),
      Rcpp::Named("threshold") = Rcpp::wrap(node.threshold),
      Rcpp::Named("lower") = Rcpp::wrap(*node.lower),
      Rcpp::Named("upper") = Rcpp::wrap(*node.upper));
  }

  SEXP wrap(const Tree<long double, int > &tree) {
    return Rcpp::List::create(
      Rcpp::Named("training_spec") = Rcpp::wrap(*tree.training_spec),
      Rcpp::Named("training_data") = Rcpp::wrap(*tree.training_data),
      Rcpp::Named("root") = Rcpp::wrap(*tree.root));
  }

  SEXP wrap(const BootstrapTree<long double, int> &tree) {
    return Rcpp::List::create(
      Rcpp::Named("training_spec") = Rcpp::wrap(*tree.training_spec),
      Rcpp::Named("training_data") = Rcpp::wrap(*tree.training_data),
      Rcpp::Named("root") = Rcpp::wrap(*tree.root));
  }

  SEXP wrap(const Forest<long double, int> &forest) {
    Rcpp::List trees(forest.trees.size());

    for (size_t i = 0; i < forest.trees.size(); i++) {
      trees[i] = wrap(*forest.trees[i]);
    }

    return Rcpp::List::create(
      Rcpp::Named("training_spec") = Rcpp::wrap(*forest.training_spec),
      Rcpp::Named("training_data") = Rcpp::wrap(*forest.training_data),
      Rcpp::Named("seed") = Rcpp::wrap(forest.seed),
      Rcpp::Named("n_threads") = Rcpp::wrap(forest.n_threads),
      Rcpp::Named("trees") = trees);
  }

  SEXP wrap(const TrainingSpec<long double, int> &spec) {
    struct TrainingSpecWrapper : public TrainingSpecVisitor<long double, int> {
      Rcpp::List result;

      void visit(const GLDATrainingSpec<long double, int> &spec) {
        result = Rcpp::wrap(spec);
      }

      void visit(const UniformGLDATrainingSpec<long double, int> &spec) {
        result = Rcpp::wrap(spec);
      }
    };

    TrainingSpecWrapper wrapper;
    spec.accept(wrapper);
    return wrapper.result;
  }

  SEXP wrap(const GLDATrainingSpec<long double, int> &spec) {
    return Rcpp::List::create(
      Rcpp::Named("strategy") = "glda",
      Rcpp::Named("lambda") = Rcpp::wrap(spec.lambda));
  }

  SEXP wrap(const UniformGLDATrainingSpec<long double, int> &spec) {
    return Rcpp::List::create(
      Rcpp::Named("strategy") = "uniform_glda",
      Rcpp::Named("n_vars") = Rcpp::wrap(spec.n_vars),
      Rcpp::Named("lambda") = Rcpp::wrap(spec.lambda));
  }

  SEXP wrap(const DataSpec<long double, int> &data) {
    return Rcpp::List::create(
      Rcpp::Named("x") = Rcpp::wrap(data.x),
      Rcpp::Named("y") = Rcpp::wrap(data.y),
      Rcpp::Named("classes") = Rcpp::wrap(data.classes));
  }

  SEXP wrap(const BootstrapDataSpec<long double, int> &data) {
    return Rcpp::List::create(
      Rcpp::Named("x") = Rcpp::wrap(data.x),
      Rcpp::Named("y") = Rcpp::wrap(data.y),
      Rcpp::Named("classes") = Rcpp::wrap(data.classes),
      Rcpp::Named("sample_indices") = Rcpp::wrap(data.sample_indices));
  }

  template<> std::unique_ptr<Node<long double, int> > as(SEXP x) {
    Rcpp::List rnode(x);

    if (rnode.containsElementNamed("value")) {
      auto resp = as<Response<long double, int> >(x);

      auto resp_ptr = std::make_unique<Response<long double, int> >(std::move(resp));
      return resp_ptr;
    }

    auto cond = as<Condition<long double, int> >(x);
    auto cond_ptr = std::make_unique<Condition<long double, int> >(std::move(cond));
    return cond_ptr;
  }

  template<> Response<long double, int> as(SEXP x) {
    Rcpp::List rresp(x);
    return Response<long double, int>(Rcpp::as<long double>(rresp["value"]));
  }

  template<> Condition<long double, int> as(SEXP x) {
    Rcpp::List rcond(x);

    auto lower = as<std::unique_ptr<Node<long double, int> > >(rcond["lower"]);
    auto upper = as<std::unique_ptr<Node<long double, int> > >(rcond["upper"]);

    return Condition<long double, int>(
      Rcpp::as<Projector<long double> >(rcond["projector"]),
      Rcpp::as<long double>(rcond["threshold"]),
      std::move(lower),
      std::move(upper));
  }

  template<> Tree<long double, int > as(SEXP x) {
    Rcpp::List rtree(x);
    Rcpp::List rtraining_spec(rtree["training_spec"]);
    Rcpp::List rtraining_data(rtree["training_data"]);

    auto root = as<Condition<long double, int> >(rtree["root"]);
    auto root_ptr = std::make_unique<Condition<long double, int> >(std::move(root));
    auto training_spec_ptr = as<std::unique_ptr<TrainingSpec<long double, int> > >(rtraining_spec);

    return Tree<long double, int >(
      std::move(root_ptr),
      std::move(training_spec_ptr),
      std::make_shared<DataSpec<long double, int> >(as<DataSpec<long double, int> >(rtraining_data)));
  }

  template<> BootstrapTree<long double, int> as(SEXP x) {
    Rcpp::List rtree(x);
    Rcpp::List rtraining_spec(rtree["training_spec"]);
    Rcpp::List rtraining_data(rtree["training_data"]);

    auto root = as<Condition<long double, int> >(rtree["root"]);
    auto root_ptr = std::make_unique<Condition<long double, int> >(std::move(root));

    auto training_spec_ptr = as<std::unique_ptr<TrainingSpec<long double, int> > >(rtraining_spec);

    return BootstrapTree<long double, int>(
      std::move(root_ptr),
      std::move(training_spec_ptr),
      std::make_shared<BootstrapDataSpec<long double, int> >(as<BootstrapDataSpec<long double, int> >(rtraining_data)));
  }

  template<> Forest<long double, int> as(SEXP x) {
    Rcpp::List rforest(x);
    Rcpp::List rtrees(rforest["trees"]);
    Rcpp::List rtraining_spec(rforest["training_spec"]);
    Rcpp::List rtraining_data(rforest["training_data"]);

    auto training_spec_ptr = as<std::unique_ptr<TrainingSpec<long double, int> > >(rtraining_spec);

    Forest<long double, int> forest(
      std::move(training_spec_ptr),
      std::make_shared<DataSpec<long double, int> >(as<DataSpec<long double, int> >(rtraining_data)),
      Rcpp::as<double>(rforest["seed"]),
      Rcpp::as<int>(rforest["n_threads"]));

    for (size_t i = 0; i < rtrees.size(); i++) {
      auto tree = as<BootstrapTree<long double, int> > (rtrees[i]);
      auto tree_ptr = std::make_unique<BootstrapTree<long double, int> > (std::move(tree));
      forest.add_tree(std::move(tree_ptr));
    }

    return forest;
  }

  template<> std::unique_ptr<TrainingSpec<long double, int> > as(SEXP x) {
    Rcpp::List rtraining_spec(x);

    std::string strategy = Rcpp::as<std::string>(rtraining_spec["strategy"]);

    if (strategy == "glda") {
      double lambda = Rcpp::as<double>(rtraining_spec["lambda"]);
      return std::make_unique<GLDATrainingSpec<long double, int> >(lambda);
    }

    if (strategy == "uniform_glda") {
      int n_vars = Rcpp::as<int>(rtraining_spec["n_vars"]);
      double lambda = Rcpp::as<double>(rtraining_spec["lambda"]);

      return std::make_unique<UniformGLDATrainingSpec<long double, int> >(n_vars, lambda);
    }

    Rcpp::stop("Unknown training strategy: %s", strategy);
  }

  template<> DataSpec<long double, int> as(SEXP x) {
    Rcpp::List rdata(x);

    std::vector<int> classes = Rcpp::as<std::vector<int> >(rdata["classes"]);

    return DataSpec<long double, int>(
      Rcpp::as<Data<long double> >(rdata["x"]),
      Rcpp::as<DataColumn<int> >(rdata["y"]),
      std::set<int>(classes.begin(), classes.end()));
  }

  template<> BootstrapDataSpec<long double, int> as(SEXP x) {
    Rcpp::List rdata(x);

    std::vector<int> classes = Rcpp::as<std::vector<int> >(rdata["classes"]);
    std::vector<int> sample_indices = Rcpp::as<std::vector<int> >(rdata["sample_indices"]);

    return BootstrapDataSpec<long double, int>(
      Rcpp::as<Data<long double> >(rdata["x"]),
      Rcpp::as<DataColumn<int> >(rdata["y"]),
      std::set<int>(classes.begin(), classes.end()),
      sample_indices);
  }
}
