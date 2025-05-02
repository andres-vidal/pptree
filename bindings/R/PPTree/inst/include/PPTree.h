#include "pptree.hpp"

#include <RcppCommon.h>

using namespace pptree;

namespace Rcpp {
  SEXP wrap(const TreeNode<float, int> &);
  SEXP wrap(const Tree<float, int > &);
  SEXP wrap(const BootstrapTree<float, int> &);
  SEXP wrap(const TreeResponse<float, int> &);
  SEXP wrap(const TreeCondition<float, int> &);
  SEXP wrap(const Forest<float, int> &);

  SEXP wrap(const TrainingSpec<float, int> &);
  SEXP wrap(const TrainingSpecGLDA<float, int> &);
  SEXP wrap(const TrainingSpecUGLDA<float, int> &);

  SEXP wrap(const SortedDataSpec<float, int> &);
  SEXP wrap(const BootstrapDataSpec<float, int> &);

  template<> std::unique_ptr<TreeNode<float, int> > as(SEXP);
  template<> Tree<float, int > as(SEXP);
  template<> BootstrapTree<float, int> as(SEXP);
  template<> TreeResponse<float, int> as(SEXP);
  template<> TreeCondition<float, int> as(SEXP);
  template<> Forest<float, int> as(SEXP);

  template<> TrainingSpecPtr<float, int> as(SEXP);

  template<> SortedDataSpec<float, int>  as(SEXP);
  template<> BootstrapDataSpec<float, int> as(SEXP);
}


#include <Rcpp.h>

namespace Rcpp {
  SEXP wrap(const TreeNode<float, int>& node) {
    struct NodeWrapper : public TreeNodeVisitor<float, int> {
      Rcpp::List result;

      void visit(const TreeCondition<float, int> &condition) {
        result = Rcpp::wrap(condition);
      }

      void visit(const TreeResponse<float, int> &response) {
        result = Rcpp::wrap(response);
      }
    };

    NodeWrapper wrapper;
    node.accept(wrapper);
    return wrapper.result;
  }

  SEXP wrap(const TreeResponse<float, int> &node) {
    return Rcpp::List::create(
      Rcpp::Named("value") = Rcpp::wrap(node.value));
  }

  SEXP wrap(const TreeCondition<float, int> &node) {
    return Rcpp::List::create(
      Rcpp::Named("projector") = Rcpp::wrap(node.projector),
      Rcpp::Named("threshold") = Rcpp::wrap(node.threshold),
      Rcpp::Named("lower")     = Rcpp::wrap(*node.lower),
      Rcpp::Named("upper")     = Rcpp::wrap(*node.upper));
  }

  SEXP wrap(const Tree<float, int > &tree) {
    return Rcpp::List::create(
      Rcpp::Named("training_data") = Rcpp::List::create(
        Rcpp::Named("x")           = Rcpp::wrap(tree.x),
        Rcpp::Named("y")           = Rcpp::wrap(tree.y),
        Rcpp::Named("classes")     = Rcpp::wrap(tree.classes)),
      Rcpp::Named("training_spec") = Rcpp::wrap(*tree.training_spec),
      Rcpp::Named("root")          = Rcpp::wrap(*tree.root));
  }

  SEXP wrap(const BootstrapTree<float, int> &tree) {
    return Rcpp::List::create(
      Rcpp::Named("training_data") = Rcpp::List::create(
        Rcpp::Named("x")           = Rcpp::wrap(tree.x),
        Rcpp::Named("y")           = Rcpp::wrap(tree.y),
        Rcpp::Named("classes")     = Rcpp::wrap(tree.classes),
        Rcpp::Named("iob_indices") = Rcpp::wrap(tree.iob_indices),
        Rcpp::Named("oob_indices") = Rcpp::wrap(tree.oob_indices)),
      Rcpp::Named("training_spec") = Rcpp::wrap(*tree.training_spec),
      Rcpp::Named("root")          = Rcpp::wrap(*tree.root));
  }

  SEXP wrap(const Forest<float, int> &forest) {
    Rcpp::List trees(forest.trees.size());

    for (size_t i = 0; i < forest.trees.size(); i++) {
      trees[i] = wrap(*forest.trees[i]);
    }

    return Rcpp::List::create(
      Rcpp::Named("training_spec") = Rcpp::wrap(*forest.training_spec),
      Rcpp::Named("training_data") = Rcpp::wrap(forest.training_data),
      Rcpp::Named("seed")          = Rcpp::wrap(forest.seed),
      Rcpp::Named("n_threads")     = Rcpp::wrap(forest.n_threads),
      Rcpp::Named("trees")         = trees);
  }

  SEXP wrap(const TrainingSpec<float, int> &spec) {
    struct TrainingSpecWrapper : public TrainingSpecVisitor<float, int> {
      Rcpp::List result;

      void visit(const TrainingSpecGLDA<float, int> &spec) {
        result = Rcpp::wrap(spec);
      }

      void visit(const TrainingSpecUGLDA<float, int> &spec) {
        result = Rcpp::wrap(spec);
      }
    };

    TrainingSpecWrapper wrapper;
    spec.accept(wrapper);
    return wrapper.result;
  }

  SEXP wrap(const TrainingSpecGLDA<float, int> &spec) {
    return Rcpp::List::create(
      Rcpp::Named("strategy") = "glda",
      Rcpp::Named("lambda")   = Rcpp::wrap(spec.lambda));
  }

  SEXP wrap(const TrainingSpecUGLDA<float, int> &spec) {
    return Rcpp::List::create(
      Rcpp::Named("strategy") = "uniform_glda",
      Rcpp::Named("n_vars")   = Rcpp::wrap(spec.n_vars),
      Rcpp::Named("lambda")   = Rcpp::wrap(spec.lambda));
  }

  SEXP wrap(const SortedDataSpec<float, int> &data) {
    return Rcpp::List::create(
      Rcpp::Named("x")       = Rcpp::wrap(data.x),
      Rcpp::Named("y")       = Rcpp::wrap(data.y),
      Rcpp::Named("classes") = Rcpp::wrap(data.classes));
  }

  SEXP wrap(const BootstrapDataSpec<float, int> &data) {
    return Rcpp::List::create(
      Rcpp::Named("x")              = Rcpp::wrap(data.x),
      Rcpp::Named("y")              = Rcpp::wrap(data.y),
      Rcpp::Named("classes")        = Rcpp::wrap(data.classes),
      Rcpp::Named("sample_indices") = Rcpp::wrap(data.sample_indices));
  }

  template<> std::unique_ptr<TreeNode<float, int> > as(SEXP x) {
    Rcpp::List rnode(x);

    if (rnode.containsElementNamed("value")) {
      return as<TreeResponse<float, int> >(x).clone();
    }

    return as<TreeCondition<float, int> >(x).clone();
  }

  template<> TreeResponse<float, int> as(SEXP x) {
    Rcpp::List rresp(x);
    return TreeResponse<float, int>(Rcpp::as<float>(rresp["value"]));
  }

  template<> TreeCondition<float, int> as(SEXP x) {
    Rcpp::List rcond(x);

    auto lower = as<std::unique_ptr<TreeNode<float, int> > >(rcond["lower"]);
    auto upper = as<std::unique_ptr<TreeNode<float, int> > >(rcond["upper"]);

    return TreeCondition<float, int>(
      Rcpp::as<Projector<float> >(rcond["projector"]),
      Rcpp::as<float>(rcond["threshold"]),
      std::move(lower),
      std::move(upper));
  }

  template<> Tree<float, int > as(SEXP x) {
    Rcpp::List rtree(x);
    Rcpp::List rtraining_spec(rtree["training_spec"]);
    Rcpp::List rtraining_data(rtree["training_data"]);

    std::vector<int> classes = as<std::vector<int> >(rtraining_data["classes"]);

    return Tree<float, int >(
      as<TreeCondition<float, int> >(rtree["root"]).clone(),
      as<TrainingSpecPtr<float, int> >(rtraining_spec),
      as<Data<float> >(rtraining_data["x"]),
      as<DataColumn<int> >(rtraining_data["y"]),
      std::set<int>(classes.begin(), classes.end()));
  }

  template<> BootstrapTree<float, int> as(SEXP x) {
    Rcpp::List rtree(x);
    Rcpp::List rtraining_spec(rtree["training_spec"]);
    Rcpp::List rtraining_data(rtree["training_data"]);

    std::vector<int> classes     = as<std::vector<int> >(rtraining_data["classes"]);
    std::vector<int> oob_indices = as<std::vector<int> >(rtraining_data["oob_indices"]);

    return BootstrapTree<float, int>(
      as<TreeCondition<float, int> >(rtree["root"]).clone(),
      as<TrainingSpecPtr<float, int> >(rtraining_spec),
      as<Data<float> >(rtraining_data["x"]),
      as<DataColumn<int> >(rtraining_data["y"]),
      std::set<int>(classes.begin(), classes.end()),
      as<std::vector<int> >(rtraining_data["iob_indices"]),
      std::set<int>(oob_indices.begin(), oob_indices.end()));
  }

  template<> Forest<float, int> as(SEXP x) {
    Rcpp::List rforest(x);
    Rcpp::List rtrees(rforest["trees"]);
    Rcpp::List rtraining_spec(rforest["training_spec"]);
    Rcpp::List rtraining_data(rforest["training_data"]);

    Forest<float, int> forest(
      as<TrainingSpecPtr<float, int> >(rtraining_spec),
      as<SortedDataSpec<float, int> >(rtraining_data),
      Rcpp::as<float>(rforest["seed"]),
      Rcpp::as<int>(rforest["n_threads"]));

    for (size_t i = 0; i < rtrees.size(); i++) {
      auto tree     = as<BootstrapTree<float, int> > (rtrees[i]);
      auto tree_ptr = std::make_unique<BootstrapTree<float, int> > (std::move(tree));
      forest.add_tree(std::move(tree_ptr));
    }

    return forest;
  }

  template<> TrainingSpecPtr<float, int> as(SEXP x) {
    Rcpp::List rtraining_spec(x);

    std::string strategy = Rcpp::as<std::string>(rtraining_spec["strategy"]);

    if (strategy == "glda") {
      float lambda = Rcpp::as<float>(rtraining_spec["lambda"]);
      return TrainingSpecGLDA<float, int>::make(lambda);
    }

    if (strategy == "uniform_glda") {
      int n_vars   = Rcpp::as<int>(rtraining_spec["n_vars"]);
      float lambda = Rcpp::as<float>(rtraining_spec["lambda"]);

      return TrainingSpecUGLDA<float, int>::make(n_vars, lambda);
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

    std::vector<int> classes        = Rcpp::as<std::vector<int> >(rdata["classes"]);
    std::vector<int> sample_indices = Rcpp::as<std::vector<int> >(rdata["sample_indices"]);

    return BootstrapDataSpec<float, int>(
      Rcpp::as<Data<float> >(rdata["x"]),
      Rcpp::as<DataColumn<int> >(rdata["y"]),
      std::set<int>(classes.begin(), classes.end()),
      sample_indices);
  }
}
