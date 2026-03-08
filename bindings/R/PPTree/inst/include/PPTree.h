#include "pptree.hpp"

#include <RcppCommon.h>

using namespace pptree;
using namespace pptree::types;
using namespace pptree::stats;
using namespace pptree::pp;

namespace Rcpp {
  SEXP wrap(const TreeNode &);
  SEXP wrap(const Tree &);
  SEXP wrap(const BootstrapTree &);
  SEXP wrap(const TreeResponse &);
  SEXP wrap(const TreeCondition &);
  SEXP wrap(const Forest &);

  SEXP wrap(const TrainingSpec &);
  SEXP wrap(const TrainingSpecGLDA &);
  SEXP wrap(const TrainingSpecUGLDA &);

  template<> std::unique_ptr<TreeNode> as(SEXP);
  template<> Tree as(SEXP);
  template<> BootstrapTree as(SEXP);
  template<> TreeResponse as(SEXP);
  template<> TreeCondition as(SEXP);
  template<> Forest as(SEXP);

  template<> TrainingSpec::Ptr as(SEXP);
}


#include <Rcpp.h>

namespace Rcpp {
  SEXP wrap(const TreeNode& node) {
    struct NodeWrapper : public TreeNodeVisitor {
      Rcpp::List result;

      void visit(const TreeCondition &condition) {
        result = Rcpp::wrap(condition);
      }

      void visit(const TreeResponse &response) {
        result = Rcpp::wrap(response);
      }
    };

    NodeWrapper wrapper;
    node.accept(wrapper);
    return wrapper.result;
  }

  SEXP wrap(const TreeResponse &node) {
    return Rcpp::List::create(
      Rcpp::Named("value") = Rcpp::wrap(node.value));
  }

  SEXP wrap(const TreeCondition &node) {
    Rcpp::IntegerVector classes(node.classes.begin(), node.classes.end());

    return Rcpp::List::create(
      Rcpp::Named("projector")      = Rcpp::wrap(node.projector),
      Rcpp::Named("threshold")      = Rcpp::wrap(node.threshold),
      Rcpp::Named("pp_index_value") = Rcpp::wrap(node.pp_index_value),
      Rcpp::Named("classes")        = classes,
      Rcpp::Named("lower")          = Rcpp::wrap(*node.lower),
      Rcpp::Named("upper")          = Rcpp::wrap(*node.upper));
  }

  SEXP wrap(const Tree &tree) {
    return Rcpp::List::create(
      Rcpp::Named("training_spec") = Rcpp::wrap(*tree.training_spec),
      Rcpp::Named("root")          = Rcpp::wrap(*tree.root));
  }

  SEXP wrap(const BootstrapTree &tree) {
    return Rcpp::List::create(
      Rcpp::Named("training_spec")  = Rcpp::wrap(*tree.training_spec),
      Rcpp::Named("root")           = Rcpp::wrap(*tree.root),
      Rcpp::Named("sample_indices") = Rcpp::wrap(tree.sample_indices));
  }

  SEXP wrap(const Forest &forest) {
    Rcpp::List trees(forest.trees.size());

    for (size_t i = 0; i < forest.trees.size(); i++) {
      trees[i] = wrap(*forest.trees[i]);
    }

    return Rcpp::List::create(
      Rcpp::Named("training_spec") = Rcpp::wrap(*forest.training_spec),
      Rcpp::Named("seed")          = Rcpp::wrap(forest.seed),
      Rcpp::Named("trees")         = trees);
  }

  SEXP wrap(const TrainingSpec &spec) {
    struct TrainingSpecWrapper : public TrainingSpecVisitor {
      Rcpp::List result;

      void visit(const TrainingSpecGLDA &spec) {
        result = Rcpp::wrap(spec);
      }

      void visit(const TrainingSpecUGLDA &spec) {
        result = Rcpp::wrap(spec);
      }
    };

    TrainingSpecWrapper wrapper;
    spec.accept(wrapper);
    return wrapper.result;
  }

  SEXP wrap(const TrainingSpecGLDA &spec) {
    return Rcpp::List::create(
      Rcpp::Named("strategy") = "glda",
      Rcpp::Named("lambda")   = Rcpp::wrap(spec.lambda));
  }

  SEXP wrap(const TrainingSpecUGLDA &spec) {
    return Rcpp::List::create(
      Rcpp::Named("strategy") = "uniform_glda",
      Rcpp::Named("n_vars")   = Rcpp::wrap(spec.n_vars),
      Rcpp::Named("lambda")   = Rcpp::wrap(spec.lambda));
  }

  template<> std::unique_ptr<TreeNode> as(SEXP x) {
    Rcpp::List rnode(x);

    if (rnode.containsElementNamed("value")) {
      return as<TreeResponse>(x).clone();
    }

    return as<TreeCondition>(x).clone();
  }

  template<> TreeResponse as(SEXP x) {
    Rcpp::List rresp(x);
    return TreeResponse(Rcpp::as<Feature>(rresp["value"]));
  }

  template<> TreeCondition as(SEXP x) {
    Rcpp::List rcond(x);

    auto lower = as<std::unique_ptr<TreeNode> >(rcond["lower"]);
    auto upper = as<std::unique_ptr<TreeNode> >(rcond["upper"]);

    std::set<Response> classes;
    if (rcond.containsElementNamed("classes")) {
      Rcpp::IntegerVector rclasses(rcond["classes"]);
      classes.insert(rclasses.begin(), rclasses.end());
    }

    Feature pp_index_value = 0;
    if (rcond.containsElementNamed("pp_index_value")) {
      pp_index_value = Rcpp::as<Feature>(rcond["pp_index_value"]);
    }

    return TreeCondition(
      Rcpp::as<Projector >(rcond["projector"]),
      Rcpp::as<Feature>(rcond["threshold"]),
      std::move(lower),
      std::move(upper),
      nullptr,
      std::move(classes),
      pp_index_value);
  }

  template<> Tree as(SEXP x) {
    Rcpp::List rtree(x);
    Rcpp::List rtraining_spec(rtree["training_spec"]);

    return Tree(
      as<TreeCondition>(rtree["root"]).clone(),
      as<TrainingSpec::Ptr>(rtraining_spec));
  }

  template<> BootstrapTree as(SEXP x) {
    Rcpp::List rtree(x);

    return BootstrapTree(
      as<TreeCondition>(rtree["root"]).clone(),
      as<TrainingSpec::Ptr>(rtree["training_spec"]),
      as<std::vector<int> >(rtree["sample_indices"]));
  }

  template<> Forest as(SEXP x) {
    Rcpp::List rforest(x);
    Rcpp::List rtrees(rforest["trees"]);
    Rcpp::List rtraining_spec(rforest["training_spec"]);

    Forest forest(
      as<TrainingSpec::Ptr>(rtraining_spec),
      Rcpp::as<Feature>(rforest["seed"]));

    for (size_t i = 0; i < rtrees.size(); i++) {
      auto tree     = as<BootstrapTree>(rtrees[i]);
      auto tree_ptr = std::make_unique<BootstrapTree>(std::move(tree));
      forest.add_tree(std::move(tree_ptr));
    }

    return forest;
  }

  template<> TrainingSpec::Ptr as(SEXP x) {
    Rcpp::List rtraining_spec(x);

    std::string strategy = Rcpp::as<std::string>(rtraining_spec["strategy"]);

    if (strategy == "glda") {
      float lambda = Rcpp::as<float>(rtraining_spec["lambda"]);
      return TrainingSpecGLDA::make(lambda);
    }

    if (strategy == "uniform_glda") {
      int n_vars   = Rcpp::as<int>(rtraining_spec["n_vars"]);
      float lambda = Rcpp::as<float>(rtraining_spec["lambda"]);

      return TrainingSpecUGLDA::make(n_vars, lambda);
    }

    Rcpp::stop("Unknown training strategy: %s", strategy);
  }
}
