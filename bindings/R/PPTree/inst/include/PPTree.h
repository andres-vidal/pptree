#include "pptree.hpp"

#include <RcppCommon.h>

using namespace pptree;

namespace Rcpp {
  SEXP wrap(const Node<float, int> &);
  SEXP wrap(const Tree<float, int > &);
  SEXP wrap(const BootstrapTree<float, int> &);
  SEXP wrap(const Response<float, int> &);
  SEXP wrap(const Condition<float, int> &);
  SEXP wrap(const Forest<float, int> &);

  SEXP wrap(const TrainingSpec<float, int> &);
  SEXP wrap(const GLDATrainingSpec<float, int> &);
  SEXP wrap(const UniformGLDATrainingSpec<float, int> &);

  SEXP wrap(const SortedDataSpec<float, int> &);
  SEXP wrap(const BootstrapDataSpec<float, int> &);

  template<> std::unique_ptr<Node<float, int> > as(SEXP);
  template<> Tree<float, int > as(SEXP);
  template<> BootstrapTree<float, int> as(SEXP);
  template<> Response<float, int> as(SEXP);
  template<> Condition<float, int> as(SEXP);
  template<> Forest<float, int> as(SEXP);

  template<> std::unique_ptr<TrainingSpec<float, int> > as(SEXP);

  template<> SortedDataSpec<float, int>  as(SEXP);
  template<> BootstrapDataSpec<float, int> as(SEXP);
}


#include <Rcpp.h>

namespace Rcpp {
  SEXP wrap(const Node<float, int>& node) {
    struct NodeWrapper : public NodeVisitor<float, int> {
      Rcpp::List result;

      void visit(const Condition<float, int> &condition) {
        result = Rcpp::wrap(condition);
      }

      void visit(const Response<float, int> &response) {
        result = Rcpp::wrap(response);
      }
    };

    NodeWrapper wrapper;
    node.accept(wrapper);
    return wrapper.result;
  }

  SEXP wrap(const Response<float, int> &node) {
    return Rcpp::List::create(
      Rcpp::Named("value") = Rcpp::wrap(node.value));
  }

  SEXP wrap(const Condition<float, int> &node) {
    return Rcpp::List::create(
      Rcpp::Named("projector") = Rcpp::wrap(node.projector),
      Rcpp::Named("threshold") = Rcpp::wrap(node.threshold),
      Rcpp::Named("lower") = Rcpp::wrap(*node.lower),
      Rcpp::Named("upper") = Rcpp::wrap(*node.upper));
  }

  SEXP wrap(const Tree<float, int > &tree) {
    return Rcpp::List::create(
      Rcpp::Named("training_spec") = Rcpp::wrap(*tree.training_spec),
      Rcpp::Named("training_data") = Rcpp::wrap(*tree.training_data),
      Rcpp::Named("root") = Rcpp::wrap(*tree.root));
  }

  SEXP wrap(const BootstrapTree<float, int> &tree) {
    return Rcpp::List::create(
      Rcpp::Named("training_spec") = Rcpp::wrap(*tree.training_spec),
      Rcpp::Named("training_data") = Rcpp::wrap(*tree.training_data),
      Rcpp::Named("root") = Rcpp::wrap(*tree.root));
  }

  SEXP wrap(const Forest<float, int> &forest) {
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

  SEXP wrap(const TrainingSpec<float, int> &spec) {
    struct TrainingSpecWrapper : public TrainingSpecVisitor<float, int> {
      Rcpp::List result;

      void visit(const GLDATrainingSpec<float, int> &spec) {
        result = Rcpp::wrap(spec);
      }

      void visit(const UniformGLDATrainingSpec<float, int> &spec) {
        result = Rcpp::wrap(spec);
      }
    };

    TrainingSpecWrapper wrapper;
    spec.accept(wrapper);
    return wrapper.result;
  }

  SEXP wrap(const GLDATrainingSpec<float, int> &spec) {
    return Rcpp::List::create(
      Rcpp::Named("strategy") = "glda",
      Rcpp::Named("lambda") = Rcpp::wrap(spec.lambda));
  }

  SEXP wrap(const UniformGLDATrainingSpec<float, int> &spec) {
    return Rcpp::List::create(
      Rcpp::Named("strategy") = "uniform_glda",
      Rcpp::Named("n_vars") = Rcpp::wrap(spec.n_vars),
      Rcpp::Named("lambda") = Rcpp::wrap(spec.lambda));
  }

  SEXP wrap(const SortedDataSpec<float, int> &data) {
    return Rcpp::List::create(
      Rcpp::Named("x") = Rcpp::wrap(data.x),
      Rcpp::Named("y") = Rcpp::wrap(data.y),
      Rcpp::Named("classes") = Rcpp::wrap(data.classes));
  }

  SEXP wrap(const BootstrapDataSpec<float, int> &data) {
    return Rcpp::List::create(
      Rcpp::Named("x") = Rcpp::wrap(data.x),
      Rcpp::Named("y") = Rcpp::wrap(data.y),
      Rcpp::Named("classes") = Rcpp::wrap(data.classes),
      Rcpp::Named("sample_indices") = Rcpp::wrap(data.sample_indices));
  }

  template<> std::unique_ptr<Node<float, int> > as(SEXP x) {
    Rcpp::List rnode(x);

    if (rnode.containsElementNamed("value")) {
      auto resp = as<Response<float, int> >(x);

      auto resp_ptr = std::make_unique<Response<float, int> >(std::move(resp));
      return resp_ptr;
    }

    auto cond = as<Condition<float, int> >(x);
    auto cond_ptr = std::make_unique<Condition<float, int> >(std::move(cond));
    return cond_ptr;
  }

  template<> Response<float, int> as(SEXP x) {
    Rcpp::List rresp(x);
    return Response<float, int>(Rcpp::as<float>(rresp["value"]));
  }

  template<> Condition<float, int> as(SEXP x) {
    Rcpp::List rcond(x);

    auto lower = as<std::unique_ptr<Node<float, int> > >(rcond["lower"]);
    auto upper = as<std::unique_ptr<Node<float, int> > >(rcond["upper"]);

    return Condition<float, int>(
      Rcpp::as<Projector<float> >(rcond["projector"]),
      Rcpp::as<float>(rcond["threshold"]),
      std::move(lower),
      std::move(upper));
  }

  template<> Tree<float, int > as(SEXP x) {
    Rcpp::List rtree(x);
    Rcpp::List rtraining_spec(rtree["training_spec"]);
    Rcpp::List rtraining_data(rtree["training_data"]);

    auto root = as<Condition<float, int> >(rtree["root"]);
    auto root_ptr = std::make_unique<Condition<float, int> >(std::move(root));
    auto training_spec_ptr = as<std::unique_ptr<TrainingSpec<float, int> > >(rtraining_spec);

    return Tree<float, int >(
      std::move(root_ptr),
      std::move(training_spec_ptr),
      std::make_shared<SortedDataSpec<float, int> >(as<SortedDataSpec<float, int> >(rtraining_data)));
  }

  template<> BootstrapTree<float, int> as(SEXP x) {
    Rcpp::List rtree(x);
    Rcpp::List rtraining_spec(rtree["training_spec"]);
    Rcpp::List rtraining_data(rtree["training_data"]);

    auto root = as<Condition<float, int> >(rtree["root"]);
    auto root_ptr = std::make_unique<Condition<float, int> >(std::move(root));

    auto training_spec_ptr = as<std::unique_ptr<TrainingSpec<float, int> > >(rtraining_spec);

    return BootstrapTree<float, int>(
      std::move(root_ptr),
      std::move(training_spec_ptr),
      std::make_shared<BootstrapDataSpec<float, int> >(as<BootstrapDataSpec<float, int> >(rtraining_data)));
  }

  template<> Forest<float, int> as(SEXP x) {
    Rcpp::List rforest(x);
    Rcpp::List rtrees(rforest["trees"]);
    Rcpp::List rtraining_spec(rforest["training_spec"]);
    Rcpp::List rtraining_data(rforest["training_data"]);

    auto training_spec_ptr = as<std::unique_ptr<TrainingSpec<float, int> > >(rtraining_spec);

    Forest<float, int> forest(
      std::move(training_spec_ptr),
      std::make_shared<SortedDataSpec<float, int> >(as<SortedDataSpec<float, int> >(rtraining_data)),
      Rcpp::as<float>(rforest["seed"]),
      Rcpp::as<int>(rforest["n_threads"]));

    for (size_t i = 0; i < rtrees.size(); i++) {
      auto tree = as<BootstrapTree<float, int> > (rtrees[i]);
      auto tree_ptr = std::make_unique<BootstrapTree<float, int> > (std::move(tree));
      forest.add_tree(std::move(tree_ptr));
    }

    return forest;
  }

  template<> std::unique_ptr<TrainingSpec<float, int> > as(SEXP x) {
    Rcpp::List rtraining_spec(x);

    std::string strategy = Rcpp::as<std::string>(rtraining_spec["strategy"]);

    if (strategy == "glda") {
      float lambda = Rcpp::as<float>(rtraining_spec["lambda"]);
      return std::make_unique<GLDATrainingSpec<float, int> >(lambda);
    }

    if (strategy == "uniform_glda") {
      int n_vars = Rcpp::as<int>(rtraining_spec["n_vars"]);
      float lambda = Rcpp::as<float>(rtraining_spec["lambda"]);

      return std::make_unique<UniformGLDATrainingSpec<float, int> >(n_vars, lambda);
    }

    Rcpp::stop("Unknown training strategy: %s", strategy);
  }

  template<> SortedDataSpec<float, int> as(SEXP x) {
    Rcpp::List rdata(x);

    std::vector<int> classes = Rcpp::as<std::vector<int> >(rdata["classes"]);

    return SortedDataSpec<float, int>(
      Rcpp::as<Data<float> >(rdata["x"]),
      Rcpp::as<DataColumn<int> >(rdata["y"]),
      std::set<int>(classes.begin(), classes.end()));
  }

  template<> BootstrapDataSpec<float, int> as(SEXP x) {
    Rcpp::List rdata(x);

    std::vector<int> classes = Rcpp::as<std::vector<int> >(rdata["classes"]);
    std::vector<int> sample_indices = Rcpp::as<std::vector<int> >(rdata["sample_indices"]);

    return BootstrapDataSpec<float, int>(
      Rcpp::as<Data<float> >(rdata["x"]),
      Rcpp::as<DataColumn<int> >(rdata["y"]),
      std::set<int>(classes.begin(), classes.end()),
      sample_indices);
  }
}
