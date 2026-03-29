#include "ppforest2.hpp"

#include <RcppCommon.h>

using namespace ppforest2;
using namespace ppforest2::types;
using namespace ppforest2::stats;
using namespace ppforest2::pp;
using namespace ppforest2::serialization;

constexpr const char* CLASS_PPRF = "pprf";
constexpr const char* CLASS_PPTR = "pptr";

namespace Rcpp {
  SEXP wrap(const TreeNode &);
  SEXP wrap(const Tree &);
  SEXP wrap(const BootstrapTree &);
  SEXP wrap(const TreeResponse &);
  SEXP wrap(const TreeCondition &);
  SEXP wrap(const Forest &);
  SEXP wrap(const Model::Ptr &);

  SEXP wrap(const TrainingSpec &);
  SEXP wrap(const VariableImportance &);
  SEXP wrap(const Export<Model::Ptr> &);

  template<> std::unique_ptr<TreeNode> as(SEXP);
  template<> Tree as(SEXP);
  template<> BootstrapTree as(SEXP);
  template<> TreeResponse as(SEXP);
  template<> TreeCondition as(SEXP);
  template<> Forest as(SEXP);

  template<> Model::Ptr as(SEXP);
  template<> TrainingSpec::Ptr as(SEXP);
}


#include <Rcpp.h>

namespace Rcpp {
  inline SEXP wrap(const TreeNode& node) {
    struct NodeWrapper : public TreeNode::Visitor {
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

  inline SEXP wrap(const TreeResponse &node) {
    Rcpp::List result = Rcpp::List::create(
      Rcpp::Named("value") = Rcpp::wrap(node.value + 1));  // C++ 0-based → R 1-based

    if (node.degenerate) {
      result["degenerate"] = true;
    }

    return result;
  }

  inline SEXP wrap(const TreeCondition &node) {
    Rcpp::IntegerVector groups(node.groups.begin(), node.groups.end());
    groups = groups + 1;  // C++ 0-based → R 1-based

    Rcpp::List result = Rcpp::List::create(
      Rcpp::Named("projector")      = Rcpp::wrap(node.projector),
      Rcpp::Named("threshold")      = Rcpp::wrap(node.threshold),
      Rcpp::Named("pp_index_value") = Rcpp::wrap(node.pp_index_value),
      Rcpp::Named("groups")        = groups,
      Rcpp::Named("lower")          = Rcpp::wrap(*node.lower),
      Rcpp::Named("upper")          = Rcpp::wrap(*node.upper));

    if (node.degenerate) {
      result["degenerate"] = true;
    }

    return result;
  }

  inline SEXP wrap(const Tree &tree) {
    Rcpp::List result = Rcpp::List::create(
      Rcpp::Named("training_spec") = Rcpp::wrap(*tree.training_spec),
      Rcpp::Named("root")          = Rcpp::wrap(*tree.root),
      Rcpp::Named("degenerate")    = tree.degenerate);
    result.attr("class") = CLASS_PPTR;
    return result;
  }

  inline SEXP wrap(const BootstrapTree &tree) {
    Rcpp::List result = Rcpp::List::create(
      Rcpp::Named("training_spec")  = Rcpp::wrap(*tree.training_spec),
      Rcpp::Named("root")           = Rcpp::wrap(*tree.root),
      Rcpp::Named("sample_indices") = Rcpp::wrap(tree.sample_indices));
    result.attr("class") = CLASS_PPTR;
    return result;
  }

  inline SEXP wrap(const Forest &forest) {
    Rcpp::List trees(forest.trees.size());

    for (size_t i = 0; i < forest.trees.size(); i++) {
      trees[i] = wrap(*forest.trees[i]);
    }

    Rcpp::List result = Rcpp::List::create(
      Rcpp::Named("training_spec") = Rcpp::wrap(*forest.training_spec),
      Rcpp::Named("seed")          = Rcpp::wrap(forest.training_spec->seed),
      Rcpp::Named("trees")         = trees,
      Rcpp::Named("degenerate")    = forest.degenerate);
    result.attr("class") = CLASS_PPRF;
    return result;
  }

  inline SEXP wrap(const Model::Ptr& model) {
    struct WrapVisitor : public Model::Visitor {
      SEXP result;
      void visit(const Tree& tree)     { result = Rcpp::wrap(tree); }
      void visit(const Forest& forest) { result = Rcpp::wrap(forest); }
    };

    WrapVisitor visitor;
    model->accept(visitor);
    return visitor.result;
  }

  inline SEXP wrap(const Export<Model::Ptr>& e) {
    struct ClassVisitor : public Model::Visitor {
      const char* cls;
      void visit(const Tree&)   { cls = CLASS_PPTR; }
      void visit(const Forest&) { cls = CLASS_PPRF; }
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
  inline Rcpp::List json_to_list(const nlohmann::json& j) {
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
  inline nlohmann::json list_to_json(const Rcpp::List& list) {
    nlohmann::json j;
    Rcpp::CharacterVector names = list.names();

    for (int i = 0; i < list.size(); i++) {
      std::string key = Rcpp::as<std::string>(names[i]);

      // display_name is a presentation-only field, not part of serialization.
      if (key == "display_name") continue;

      switch (TYPEOF(list[i])) {
        case STRSXP:
          j[key] = Rcpp::as<std::string>(list[i]);
          break;
        case INTSXP:
          j[key] = Rcpp::as<int>(list[i]);
          break;
        case REALSXP:
          j[key] = Rcpp::as<float>(list[i]);
          break;
        case LGLSXP:
          j[key] = Rcpp::as<bool>(list[i]);
          break;
        default:
          break;
      }
    }
    return j;
  }

  inline SEXP wrap(const TrainingSpec &spec) {
    nlohmann::json pp_json, dr_json, sr_json;
    spec.pp_strategy->to_json(pp_json);
    spec.dr_strategy->to_json(dr_json);
    spec.sr_strategy->to_json(sr_json);

    pp_json["display_name"] = spec.pp_strategy->display_name();
    dr_json["display_name"] = spec.dr_strategy->display_name();
    sr_json["display_name"] = spec.sr_strategy->display_name();

    return Rcpp::List::create(
      Rcpp::Named("pp")          = json_to_list(pp_json),
      Rcpp::Named("dr")          = json_to_list(dr_json),
      Rcpp::Named("sr")          = json_to_list(sr_json),
      Rcpp::Named("size")        = spec.size,
      Rcpp::Named("seed")        = spec.seed,
      Rcpp::Named("threads")   = spec.threads,
      Rcpp::Named("max_retries") = spec.max_retries);
  }

  inline SEXP wrap(const VariableImportance &vi) {
    Rcpp::List list;
    if (vi.scale.size() > 0)                list["scale"]       = Rcpp::wrap(vi.scale);
    if (vi.projections.size() > 0)          list["projections"] = Rcpp::wrap(vi.projections);
    if (vi.weighted_projections.size() > 0) list["weighted"]    = Rcpp::wrap(vi.weighted_projections);
    if (vi.permuted.size() > 0)             list["permuted"]    = Rcpp::wrap(vi.permuted);
    return list;
  }

  template<> inline std::unique_ptr<TreeNode> as(SEXP x) {
    Rcpp::List rnode(x);

    if (rnode.containsElementNamed("value")) {
      return std::make_unique<TreeResponse>(
        Rcpp::as<Feature>(rnode["value"]) - 1);  // R 1-based → C++ 0-based
    }

    auto lower = as<std::unique_ptr<TreeNode> >(rnode["lower"]);
    auto upper = as<std::unique_ptr<TreeNode> >(rnode["upper"]);

    std::set<Response> groups;
    if (rnode.containsElementNamed("groups")) {
      Rcpp::IntegerVector rgroups(rnode["groups"]);
      for (auto g : rgroups) groups.insert(g - 1);  // R 1-based → C++ 0-based
    }

    Feature pp_index_value = 0;
    if (rnode.containsElementNamed("pp_index_value")) {
      pp_index_value = Rcpp::as<Feature>(rnode["pp_index_value"]);
    }

    return TreeCondition::make(
      Rcpp::as<Projector >(rnode["projector"]),
      Rcpp::as<Feature>(rnode["threshold"]),
      std::move(lower),
      std::move(upper),
      std::move(groups),
      pp_index_value);
  }

  template<> inline TreeResponse as(SEXP x) {
    Rcpp::List rresp(x);
    return TreeResponse(Rcpp::as<Feature>(rresp["value"]) - 1);  // R 1-based → C++ 0-based
  }

  template<> inline TreeCondition as(SEXP x) {
    Rcpp::List rcond(x);

    auto lower = as<std::unique_ptr<TreeNode> >(rcond["lower"]);
    auto upper = as<std::unique_ptr<TreeNode> >(rcond["upper"]);

    std::set<Response> groups;
    if (rcond.containsElementNamed("groups")) {
      Rcpp::IntegerVector rgroups(rcond["groups"]);
      for (auto g : rgroups) groups.insert(g - 1);  // R 1-based → C++ 0-based
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
      std::move(groups),
      pp_index_value);
  }

  template<> inline Tree as(SEXP x) {
    Rcpp::List rtree(x);

    return Tree(
      as<std::unique_ptr<TreeNode> >(rtree["root"]),
      as<TrainingSpec::Ptr>(rtree["training_spec"]));
  }

  template<> inline BootstrapTree as(SEXP x) {
    Rcpp::List rtree(x);

    return BootstrapTree(
      as<std::unique_ptr<TreeNode> >(rtree["root"]),
      as<TrainingSpec::Ptr>(rtree["training_spec"]),
      as<std::vector<int> >(rtree["sample_indices"]));
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

  template<> inline TrainingSpec::Ptr as(SEXP x) {
    Rcpp::List r_training_spec(x);

    return TrainingSpec::make(
      pp::PPStrategy::from_json(list_to_json(Rcpp::List(r_training_spec["pp"]))),
      dr::DRStrategy::from_json(list_to_json(Rcpp::List(r_training_spec["dr"]))),
      sr::SRStrategy::from_json(list_to_json(Rcpp::List(r_training_spec["sr"]))),
      Rcpp::as<int>(r_training_spec["size"]),
      Rcpp::as<int>(r_training_spec["seed"]),
      Rcpp::as<int>(r_training_spec["threads"]),
      Rcpp::as<int>(r_training_spec["max_retries"]));
  }
}
