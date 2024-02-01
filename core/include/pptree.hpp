#include "pp.hpp"

namespace pptree {
  template<typename T>
  using Threshold = T;

  template<typename T, typename R >
  struct Node {
    pp::Projector<T> projector;
    pptree::Threshold<T> threshold;
    pptree::Node<T, R> *left;
    pptree::Node<T, R> *right;
    R response;
  };

  template<typename T, typename R >
  struct Tree {
    Node<T, R> root;
  };

  template<typename T, typename R>
  Tree<T, R> train(
    stats::Data<T>       data,
    stats::DataColumn<R> groups,
    pp::PPStrategy<T, R> pp_strategy);


  template<typename T, typename R>
  Tree<T, R> train_lda(
    stats::Data<T>       data,
    stats::DataColumn<R> groups);

  template<typename T, typename R>
  R predict(
    stats::DataColumn<T> data,
    Tree<T, R>           tree);

  template<typename T, typename R>
  stats::DataColumn<R> predict(
    stats::Data<T> data,
    Tree<T, R>     tree);
}
