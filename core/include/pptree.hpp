#include "pp.hpp"
#include <nlohmann/json.hpp>

namespace pptree {
  inline namespace pp { using namespace ::pp; }
  inline namespace stats { using namespace ::stats; }
  template<typename T>
  using Threshold = T;
  using json = nlohmann::json;

  template<typename T, typename R >
  struct Node {
    virtual ~Node() = default;
    virtual R response() const = 0;
    virtual R predict(DataColumn<T> data) const = 0;
    virtual std::string to_string() const = 0;
  };

  template<typename T, typename R >
  struct Condition : public Node<T, R> {
    Projector<T> projector;
    Threshold<T> threshold;
    Node<T, R> *lower = nullptr;
    Node<T, R> *upper = nullptr;

    Condition(
      Projector<T> projector,
      Threshold<T> threshold,
      Node<T, R> *lower,
      Node<T, R> *upper)
      : projector(projector), threshold(threshold), lower(lower), upper(upper) {
    }

    ~Condition() {
      delete lower;
      delete upper;
    }

    R response() const override {
      throw std::runtime_error("Condition response is undefined.");
    }

    R predict(DataColumn<T> data) const override {
      T projected_data = project((Data<T>)data, projector).value();

      if (projected_data < threshold) {
        return lower->predict(data);
      } else {
        return upper->predict(data);
      }
    }

    std::string to_string() const override {
      const Eigen::IOFormat fmt(Eigen::StreamPrecision, 0, ",", "\n");

      std::stringstream stream;
      stream << "{"
             << "\"projector\":[" << projector.transpose().format(fmt) << "],"
             << "\"threshold\":" << threshold << ","
             << "\"lower\":" << lower->to_string() << ","
             << "\"upper\":" << upper->to_string()
             << "}";
      return stream.str();
    }
  };

  template<typename T, typename R >
  struct Response : public Node<T, R> {
    R value;

    Response(R value) : value(value) {
    }

    R response() const override {
      return value;
    }

    R predict(DataColumn<T> data) const override {
      return response();
    }

    std::string to_string() const override {
      std::stringstream stream;
      stream << "{\"value\":" << value << "}";
      return stream.str();
    }
  };

  template<typename T, typename R >
  struct Tree {
    Condition<T, R> *root;

    Tree(Condition<T, R> *root) : root(root) {
    }

    ~Tree() {
      delete root;
    }

    R predict(DataColumn<T> data) const {
      return root->predict(data);
    }

    DataColumn<R> predict(Data<T> data) const {
      DataColumn<R> predictions(data.rows());

      for (int i = 0; i < data.rows(); i++) {
        predictions(i) = predict((DataColumn<T>)data.row(i));
      }

      return predictions;
    }

    std::string to_string() const {
      std::stringstream stream;
      stream << "{\"root\":" << root->to_string() << "}";
      return stream.str();
    }
  };

  template<typename T, typename R>
  Tree<T, R> train(
    stats::Data<T>       data,
    stats::DataColumn<R> groups,
    pp::PPStrategy<T, R> pp_strategy);


  template<typename T, typename R>
  std::ostream& operator<<(std::ostream & ostream, const Tree<T, R>& tree) {
    json json_tree = json::parse(tree.to_string());
    return ostream << json_tree.dump(2, ' ', true);
  }

  template<typename T, typename R>
  std::ostream& operator<<(std::ostream & ostream, const Node<T, R>& node) {
    json json_node = json::parse(node.to_string());
    return ostream << json_node.dump(2, ' ', true);
  }

  template<typename T, typename R>
  std::ostream& operator<<(std::ostream & ostream, const Condition<T, R>& condition) {
    json json_condition = json::parse(condition.to_string());
    return ostream << json_condition.dump(2, ' ', true);
  }

  template<typename T, typename R>
  std::ostream& operator<<(std::ostream & ostream, const Response<T, R>& response) {
    json json_response = json::parse(response.to_string());
    return ostream << json_response.dump(2, ' ', true);
  }
}
