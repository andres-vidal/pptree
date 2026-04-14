#include "ppforest2.hpp"

#include <RcppCommon.h>

using namespace ppforest2;
using namespace ppforest2::types;
using namespace ppforest2::stats;
using namespace ppforest2::serialization;
using namespace ppforest2::pp;

constexpr char const* CLASS_PPRF = "pprf";
constexpr char const* CLASS_PPTR = "pptr";

namespace Rcpp {
  SEXP wrap(TreeNode const&);
  SEXP wrap(Tree const&);
  SEXP wrap(BootstrapTree const&);
  SEXP wrap(TreeLeaf const&);
  SEXP wrap(TreeBranch const&);
  SEXP wrap(Forest const&);
  SEXP wrap(Model::Ptr const&);

  SEXP wrap(TrainingSpec const&);
  SEXP wrap(VariableImportance const&);
  SEXP wrap(Export<Model::Ptr> const&);

  template<> std::unique_ptr<TreeNode> as(SEXP);
  template<> Tree as(SEXP);
  template<> BootstrapTree as(SEXP);
  template<> TreeLeaf as(SEXP);
  template<> TreeBranch as(SEXP);
  template<> Forest as(SEXP);

  template<> Model::Ptr as(SEXP);
  template<> TrainingSpec::Ptr as(SEXP);
}


#include <Rcpp.h>

namespace Rcpp {
  inline SEXP wrap(TreeNode const& node) {
    struct NodeWrapper : public TreeNode::Visitor {
      Rcpp::List result;

      void visit(TreeBranch const& condition) { result = Rcpp::wrap(condition); }

      void visit(TreeLeaf const& response) { result = Rcpp::wrap(response); }
    };

    NodeWrapper wrapper;
    node.accept(wrapper);
    return wrapper.result;
  }

  inline SEXP wrap(TreeLeaf const& node) {
    Rcpp::List result =
        Rcpp::List::create(Rcpp::Named("value") = Rcpp::wrap(node.value + 1)); // C++ 0-based → R 1-based

    if (node.degenerate) {
      result["degenerate"] = true;
    }

    return result;
  }

  inline SEXP wrap(TreeBranch const& node) {
    Rcpp::IntegerVector groups(node.groups.begin(), node.groups.end());
    groups = groups + 1; // C++ 0-based → R 1-based

    Rcpp::List result = Rcpp::List::create(
        Rcpp::Named("projector")      = Rcpp::wrap(node.projector),
        Rcpp::Named("cutpoint")       = Rcpp::wrap(node.cutpoint),
        Rcpp::Named("pp_index_value") = Rcpp::wrap(node.pp_index_value),
        Rcpp::Named("groups")         = groups,
        Rcpp::Named("lower")          = Rcpp::wrap(*node.lower),
        Rcpp::Named("upper")          = Rcpp::wrap(*node.upper)
    );

    if (node.degenerate) {
      result["degenerate"] = true;
    }

    return result;
  }

  inline SEXP wrap(Tree const& tree) {
    Rcpp::List result = Rcpp::List::create(
        Rcpp::Named("training_spec") = Rcpp::wrap(*tree.training_spec),
        Rcpp::Named("root")          = Rcpp::wrap(*tree.root),
        Rcpp::Named("degenerate")    = tree.degenerate
    );
    result.attr("class") = CLASS_PPTR;
    return result;
  }

  inline SEXP wrap(BootstrapTree const& tree) {
    Rcpp::List result = Rcpp::List::create(
        Rcpp::Named("training_spec")  = Rcpp::wrap(*tree.training_spec),
        Rcpp::Named("root")           = Rcpp::wrap(*tree.root),
        Rcpp::Named("sample_indices") = Rcpp::wrap(tree.sample_indices)
    );
    result.attr("class") = CLASS_PPTR;
    return result;
  }

  inline SEXP wrap(Forest const& forest) {
    Rcpp::List trees(forest.trees.size());

    for (size_t i = 0; i < forest.trees.size(); i++) {
      trees[i] = wrap(*forest.trees[i]);
    }

    Rcpp::List result = Rcpp::List::create(
        Rcpp::Named("training_spec") = Rcpp::wrap(*forest.training_spec),
        Rcpp::Named("seed")          = Rcpp::wrap(forest.training_spec->seed),
        Rcpp::Named("trees")         = trees,
        Rcpp::Named("degenerate")    = forest.degenerate
    );
    result.attr("class") = CLASS_PPRF;
    return result;
  }

  inline SEXP wrap(Model::Ptr const& model) {
    struct WrapVisitor : public Model::Visitor {
      SEXP result;
      void visit(Tree const& tree) { result = Rcpp::wrap(tree); }
      void visit(Forest const& forest) { result = Rcpp::wrap(forest); }
    };

    WrapVisitor visitor;
    model->accept(visitor);
    return visitor.result;
  }

  inline SEXP wrap(Export<Model::Ptr> const& e) {
    struct ClassVisitor : public Model::Visitor {
      char const* cls;
      void visit(Tree const&) { cls = CLASS_PPTR; }
      void visit(Forest const&) { cls = CLASS_PPRF; }
    };

    Rcpp::List result = Rcpp::wrap(e.model);

    result["groups"] = Rcpp::wrap(e.groups);
    if (e.model->training_spec)
      result["seed"] = e.model->training_spec->seed;
    if (e.variable_importance)
      result["vi"] = Rcpp::wrap(*e.variable_importance);
    if (e.oob_error)
      result["oob_error"] = *e.oob_error;

    ClassVisitor cv;
    e.model->accept(cv);
    result.attr("class") = cv.cls;

    return result;
  }

  /**
   * Convert a flat JSON object to an Rcpp named list.
   *
   * Supports string, number (int/float), and boolean values.
   * Used to generically wrap strategy JSON so that adding a new
   * strategy only requires implementing to_json() — no changes here.
   */
  inline Rcpp::List json_to_list(nlohmann::json const& j) {
    Rcpp::List result;

    for (auto& [key, val] : j.items()) {
      if (val.is_string()) {
        result[key] = val.get<std::string>();
      } else if (val.is_number_integer()) {
        result[key] = val.get<int>();
      } else if (val.is_number_float()) {
        result[key] = val.get<float>();
      } else if (val.is_boolean()) {
        result[key] = val.get<bool>();
      }
    }
    return result;
  }

  /**
   * Convert an Rcpp named list to a flat JSON object.
   *
   * Inverse of json_to_list(). Supports string, integer, and real values.
   * Used to generically unwrap strategy R lists so that adding a new
   * strategy only requires implementing from_json() — no changes here.
   */
  inline nlohmann::json list_to_json(Rcpp::List const& list) {
    nlohmann::json j;
    Rcpp::CharacterVector names = list.names();

    for (int i = 0; i < list.size(); i++) {
      std::string key = Rcpp::as<std::string>(names[i]);

      // display_name is a presentation-only field, not part of serialization.
      if (key == "display_name")
        continue;

      switch (TYPEOF(list[i])) {
        case STRSXP: j[key] = Rcpp::as<std::string>(list[i]); break;
        case INTSXP: j[key] = Rcpp::as<int>(list[i]); break;
        case REALSXP: j[key] = Rcpp::as<float>(list[i]); break;
        case LGLSXP: j[key] = Rcpp::as<bool>(list[i]); break;
        default: break;
      }
    }
    return j;
  }

  /**
   * Convert a TrainingSpec to an R list.
   *
   * C++ TrainingSpec → JSON → R list: uses the strategy registry so new
   * strategies work here automatically without changes to this file.
   */
  /** Convert a strategy shared_ptr to an R list (JSON + display_name). */
  template<typename T, std::enable_if_t<std::is_base_of_v<Strategy<T>, T>, int> = 0>
  inline SEXP wrap(std::shared_ptr<T> const& strategy) {
    auto j            = strategy->to_json();
    j["display_name"] = strategy->display_name();
    return json_to_list(j);
  }

  inline SEXP wrap(TrainingSpec const& spec) {
    return Rcpp::List::create(
        Rcpp::Named("pp")          = wrap(spec.pp),
        Rcpp::Named("vars")        = wrap(spec.vars),
        Rcpp::Named("cutpoint")    = wrap(spec.cutpoint),
        Rcpp::Named("stop")        = wrap(spec.stop),
        Rcpp::Named("binarize")    = wrap(spec.binarize),
        Rcpp::Named("partition")   = wrap(spec.partition),
        Rcpp::Named("leaf")        = wrap(spec.leaf),
        Rcpp::Named("size")        = spec.size,
        Rcpp::Named("seed")        = spec.seed,
        Rcpp::Named("threads")     = spec.threads,
        Rcpp::Named("max_retries") = spec.max_retries
    );
  }

  inline SEXP wrap(VariableImportance const& vi) {
    Rcpp::List list;
    if (vi.scale.size() > 0)
      list["scale"] = Rcpp::wrap(vi.scale);
    if (vi.projections.size() > 0)
      list["projections"] = Rcpp::wrap(vi.projections);
    if (vi.weighted_projections.size() > 0)
      list["weighted"] = Rcpp::wrap(vi.weighted_projections);
    if (vi.permuted.size() > 0)
      list["permuted"] = Rcpp::wrap(vi.permuted);
    return list;
  }

  template<> inline std::unique_ptr<TreeNode> as(SEXP x) {
    Rcpp::List rnode(x);

    if (rnode.containsElementNamed("value")) {
      return std::make_unique<TreeLeaf>(Rcpp::as<Feature>(rnode["value"]) - 1); // R 1-based → C++ 0-based
    }

    auto lower = as<std::unique_ptr<TreeNode>>(rnode["lower"]);
    auto upper = as<std::unique_ptr<TreeNode>>(rnode["upper"]);

    std::set<Outcome> groups;
    if (rnode.containsElementNamed("groups")) {
      Rcpp::IntegerVector rgroups(rnode["groups"]);
      for (auto g : rgroups)
        groups.insert(g - 1); // R 1-based → C++ 0-based
    }

    Feature pp_index_value = 0;
    if (rnode.containsElementNamed("pp_index_value")) {
      pp_index_value = Rcpp::as<Feature>(rnode["pp_index_value"]);
    }

    return TreeBranch::make(
        Rcpp::as<Projector>(rnode["projector"]),
        Rcpp::as<Feature>(rnode["cutpoint"]),
        std::move(lower),
        std::move(upper),
        std::move(groups),
        pp_index_value
    );
  }

  template<> inline TreeLeaf as(SEXP x) {
    Rcpp::List rresp(x);
    return TreeLeaf(Rcpp::as<Feature>(rresp["value"]) - 1); // R 1-based → C++ 0-based
  }

  template<> inline TreeBranch as(SEXP x) {
    Rcpp::List rcond(x);

    auto lower = as<std::unique_ptr<TreeNode>>(rcond["lower"]);
    auto upper = as<std::unique_ptr<TreeNode>>(rcond["upper"]);

    std::set<Outcome> groups;
    if (rcond.containsElementNamed("groups")) {
      Rcpp::IntegerVector rgroups(rcond["groups"]);
      for (auto g : rgroups)
        groups.insert(g - 1); // R 1-based → C++ 0-based
    }

    Feature pp_index_value = 0;
    if (rcond.containsElementNamed("pp_index_value")) {
      pp_index_value = Rcpp::as<Feature>(rcond["pp_index_value"]);
    }

    return TreeBranch(
        Rcpp::as<Projector>(rcond["projector"]),
        Rcpp::as<Feature>(rcond["cutpoint"]),
        std::move(lower),
        std::move(upper),
        std::move(groups),
        pp_index_value
    );
  }

  template<> inline Tree as(SEXP x) {
    Rcpp::List rtree(x);

    return Tree(as<std::unique_ptr<TreeNode>>(rtree["root"]), as<TrainingSpec::Ptr>(rtree["training_spec"]));
  }

  template<> inline BootstrapTree as(SEXP x) {
    Rcpp::List rtree(x);

    return BootstrapTree(
        as<std::unique_ptr<TreeNode>>(rtree["root"]),
        as<TrainingSpec::Ptr>(rtree["training_spec"]),
        as<std::vector<int>>(rtree["sample_indices"])
    );
  }

  template<> inline Forest as(SEXP x) {
    Rcpp::List rforest(x);
    Rcpp::List rtrees(rforest["trees"]);
    Rcpp::List rtraining_spec(rforest["training_spec"]);

    Forest forest(as<TrainingSpec::Ptr>(rtraining_spec));

    for (size_t i = 0; i < rtrees.size(); i++) {
      auto tree     = as<BootstrapTree>(rtrees[i]);
      auto tree_ptr = std::make_unique<BootstrapTree>(std::move(tree));
      forest.add_tree(std::move(tree_ptr));
    }

    return forest;
  }

  template<> inline Model::Ptr as(SEXP x) {
    if (Rcpp::RObject(x).inherits(CLASS_PPRF)) {
      return std::make_shared<Forest>(as<Forest>(x));
    }
    return std::make_shared<Tree>(as<Tree>(x));
  }

  /**
   * Convert an R list to a TrainingSpec.
   *
   * R list → JSON → C++ strategy: uses the strategy registry so new
   * strategies work here automatically without changes to this file.
   *
   */
  template<> inline TrainingSpec::Ptr as(SEXP x) {
    Rcpp::List r_training_spec(x);

    auto pp        = pp::ProjectionPursuit::from_json(list_to_json(Rcpp::List(r_training_spec["pp"])));
    auto vars      = vars::VariableSelection::from_json(list_to_json(Rcpp::List(r_training_spec["vars"])));
    auto cutpoint  = cutpoint::SplitCutpoint::from_json(list_to_json(Rcpp::List(r_training_spec["cutpoint"])));
    auto stop      = stop::StopRule::from_json(list_to_json(Rcpp::List(r_training_spec["stop"])));
    auto binarize  = binarize::Binarization::from_json(list_to_json(Rcpp::List(r_training_spec["binarize"])));
    auto partition = partition::StepPartition::from_json(list_to_json(Rcpp::List(r_training_spec["partition"])));
    auto leaf      = leaf::LeafStrategy::from_json(list_to_json(Rcpp::List(r_training_spec["leaf"])));

    return TrainingSpec::builder()
        .pp(std::move(pp))
        .vars(std::move(vars))
        .cutpoint(std::move(cutpoint))
        .stop(std::move(stop))
        .binarize(std::move(binarize))
        .partition(std::move(partition))
        .leaf(std::move(leaf))
        .size(Rcpp::as<int>(r_training_spec["size"]))
        .seed(Rcpp::as<int>(r_training_spec["seed"]))
        .threads(Rcpp::as<int>(r_training_spec["threads"]))
        .max_retries(Rcpp::as<int>(r_training_spec["max_retries"]))
        .make();
  }
}
