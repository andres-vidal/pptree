#include "pptree.hpp"
#include "pptreeio.hpp"
#include <gtest/gtest.h>

#include <iostream>

using namespace pptree;

TEST(ResponseEquals, true_case) {
  Response<long double, int> r1(1);
  Response<long double, int> r2(1);

  ASSERT_TRUE(r1 == r2);
}

TEST(ResponseEquals, false_case) {
  Response<long double, int> r1(1);
  Response<long double, int> r2(2);

  ASSERT_FALSE(r1 == r2);
}

TEST(ConditionEquals, true_case) {
  Condition<long double, int> c1(
    as_projector<long double>({ 1.0, 2.0 }),
    3.0,
    std::make_unique<Response<long double, int> >(1),
    std::make_unique<Response<long double, int> >(2));

  Condition<long double, int> c2(
    as_projector<long double>({ 1.0, 2.0 }),
    3.0,
    std::make_unique<Response<long double, int> >(1),
    std::make_unique<Response<long double, int> >(2));

  ASSERT_TRUE(c1 == c2);
}

TEST(ConditionEquals, true_case_collinear_projectors) {
  Condition<long double, int> c1(
    as_projector<long double>({ 1.0, 1.0 }),
    3.0,
    std::make_unique<Response<long double, int> >(1),
    std::make_unique<Response<long double, int> >(2));

  Condition<long double, int> c2(
    as_projector<long double>({ 2.0, 2.0 }),
    3.0,
    std::make_unique<Response<long double, int> >(1),
    std::make_unique<Response<long double, int> >(2));

  ASSERT_TRUE(c1 == c2);
}

TEST(ConditionEquals, true_case_approximate_thresholds) {
  Condition<long double, int> c1(
    as_projector<long double>({ 1.0, 2.0 }),
    3.0,
    std::make_unique<Response<long double, int> >(1),
    std::make_unique<Response<long double, int> >(2));

  Condition<long double, int> c2(
    as_projector<long double>({ 1.0, 2.0 }),
    3.000000000000001,
    std::make_unique<Response<long double, int> >(1),
    std::make_unique<Response<long double, int> >(2));

  ASSERT_TRUE(c1 == c2);
}

TEST(ConditionEquals, false_case_non_collinear_projectors) {
  Condition<long double, int> c1(
    as_projector<long double>({ 1.0, 2.0 }),
    3.0,
    std::make_unique<Response<long double, int> >(1),
    std::make_unique<Response<long double, int> >(2));

  Condition<long double, int> c2(
    as_projector<long double>({ 2.0, 3.0 }),
    3.0,
    std::make_unique<Response<long double, int> >(1),
    std::make_unique<Response<long double, int> >(2));

  ASSERT_FALSE(c1 == c2);
}

TEST(ConditionEquals, false_case_different_thresholds) {
  Condition<long double, int> c1(
    as_projector<long double>({ 1.0, 2.0 }),
    3.0,
    std::make_unique<Response<long double, int> >(1),
    std::make_unique<Response<long double, int> >(2));

  Condition<long double, int> c2(
    as_projector<long double>({ 1.0, 2.0 }),
    4.0,
    std::make_unique<Response<long double, int> >(1),
    std::make_unique<Response<long double, int> >(2));

  ASSERT_FALSE(c1 == c2);
}

TEST(ConditionEquals, false_case_different_responses) {
  Condition<long double, int> c1(
    as_projector<long double>({ 1.0, 2.0 }),
    3.0,
    std::make_unique<Response<long double, int> >(1),
    std::make_unique<Response<long double, int> >(2));

  Condition<long double, int> c2(
    as_projector<long double>({ 1.0, 2.0 }),
    3.0,
    std::make_unique<Response<long double, int> >(1),
    std::make_unique<Response<long double, int> >(3));

  ASSERT_FALSE(c1 == c2);
}

TEST(ConditionEquals, false_case_different_structures) {
  Condition<long double, int> c1(
    as_projector<long double>({ 1.0, 2.0 }),
    3.0,
    std::make_unique<Response<long double, int> >(1),
    std::make_unique<Response<long double, int> >(2));

  Condition<long double, int> c2(
    as_projector<long double>({ 1.0, 2.0 }),
    3.0,
    std::make_unique<Response<long double, int> >(1),
    std::make_unique<Condition<long double, int> >(
      as_projector<long double>({ 1.0, 2.0 }),
      3.0,
      std::make_unique<Response<long double, int> >(1),
      std::make_unique<Response<long double, int> >(2)));

  ASSERT_FALSE(c1 == c2);
}

TEST(TreeEquals, true_case) {
  Tree<long double, int> t1(
    std::make_unique<Condition<long double, int> >(
      as_projector<long double>({ 1.0, 2.0 }),
      3.0,
      std::make_unique<Response<long double, int> >(1),
      std::make_unique<Condition<long double, int> >(
        as_projector<long double>({ 1.0, 2.0 }),
        3.0,
        std::make_unique<Response<long double, int> >(1),
        std::make_unique<Response<long double, int> >(2))),
    TrainingSpec<long double, int>::lda());

  Tree<long double, int> t2(
    std::make_unique<Condition<long double, int> >(
      as_projector<long double>({ 1.0, 2.0 }),
      3.0,
      std::make_unique<Response<long double, int> >(1),
      std::make_unique<Condition<long double, int> >(
        as_projector<long double>({ 1.0, 2.0 }),
        3.0,
        std::make_unique<Response<long double, int> >(1),
        std::make_unique<Response<long double, int> >(2))),
    TrainingSpec<long double, int>::lda());

  ASSERT_TRUE(t1 == t2);
}

TEST(TreeEquals, false_case) {
  Tree<long double, int> t1(
    std::make_unique<Condition<long double, int> >(
      as_projector<long double>({ 1.0, 2.0 }),
      3.0,
      std::make_unique<Response<long double, int> >(1),
      std::make_unique<Response<long double, int> >(2)),
    TrainingSpec<long double, int>::lda());

  Tree<long double, int> t2(
    std::make_unique<Condition<long double, int> >(
      as_projector<long double>({ 1.0, 2.0 }),
      3.0,
      std::make_unique<Response<long double, int> >(1),
      std::make_unique<Response<long double, int> >(3)),
    TrainingSpec<long double, int>::lda());

  ASSERT_FALSE(t1 == t2);
}

TEST(PPTreeTrainLDA, univariate_two_groups) {
  Data<long double> data(10, 1);
  data <<
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2;

  DataColumn<int> groups(10, 1);
  groups <<
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1;

  Tree<long double, int> result = pptree::train_glda(
    data,
    groups,
    0);

  Tree<long double, int> expect = Tree<long double, int>(
    std::make_unique<Condition<long double, int> >(
      as_projector<long double>({ 1.0 }),
      1.5,
      std::make_unique<Response<long double, int> >(0),
      std::make_unique<Response<long double, int> >(1)),
    TrainingSpec<long double, int>::lda());

  ASSERT_EQ(expect, result);
}

TEST(PPTreeTrainLDA, univariate_three_groups) {
  Data<long double> data(15, 1);
  data <<
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
    3, 3, 3, 3, 3;

  DataColumn<int> groups(15, 1);
  groups <<
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2;

  Tree<long double, int> result = pptree::train_glda(
    data,
    groups,
    0);

  Tree<long double, int> expect = Tree<long double, int>(
    std::make_unique<Condition<long double, int> >(
      as_projector<long double>({ 1.0 }),
      1.75,
      std::make_unique<Response<long double, int> >(0),
      std::make_unique<Condition<long double, int> >(
        as_projector<long double>({ 1.0 }),
        2.5,
        std::make_unique<Response<long double, int> >(1),
        std::make_unique<Response<long double, int> >(2))),
    TrainingSpec<long double, int>::lda());


  ASSERT_EQ(expect, result);
}

TEST(PPTreeTrainLDA, multivariate_two_groups) {
  Data<long double> data(10, 4);
  data <<
    1, 0, 1, 1,
    1, 1, 0, 0,
    1, 0, 0, 1,
    1, 1, 1, 1,
    4, 0, 0, 1,
    4, 0, 0, 2,
    4, 0, 0, 3,
    4, 1, 0, 1,
    4, 0, 1, 1,
    4, 0, 1, 2;

  DataColumn<int> groups(10);
  groups <<
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1;

  Tree<long double, int> result = train_glda(
    data,
    groups,
    0);

  Tree<long double, int> expect = Tree<long double, int>(
    std::make_unique<Condition<long double, int> >(
      as_projector<long double>({ 1.0, 0.0, 0.0, 0.0 }),
      2.5,
      std::make_unique<Response<long double, int> >(0),
      std::make_unique<Response<long double, int> >(1)
      ),
    TrainingSpec<long double, int>::lda()
    );

  ASSERT_EQ(expect, result);
}

TEST(PPTreeTrainLDA, multivariate_three_groups) {
  Data<long double> data(30, 5);
  data <<
    1, 0, 1, 1, 1,
    1, 0, 1, 0, 0,
    1, 0, 0, 0, 1,
    1, 0, 1, 2, 1,
    1, 0, 0, 1, 1,
    1, 1, 1, 1, 0,
    1, 0, 0, 2, 1,
    1, 0, 1, 1, 2,
    1, 0, 0, 2, 0,
    1, 0, 2, 1, 0,
    2, 5, 0, 0, 1,
    2, 5, 0, 0, 2,
    3, 5, 1, 0, 2,
    2, 5, 1, 0, 1,
    2, 5, 0, 1, 1,
    2, 5, 0, 1, 2,
    2, 5, 2, 1, 1,
    2, 5, 1, 1, 1,
    2, 5, 1, 1, 2,
    2, 5, 2, 1, 2,
    2, 5, 1, 2, 1,
    2, 5, 2, 1, 1,
    9, 8, 0, 0, 1,
    9, 8, 0, 0, 2,
    9, 8, 1, 0, 2,
    9, 8, 1, 0, 1,
    9, 8, 0, 1, 1,
    9, 8, 0, 1, 2,
    9, 8, 2, 1, 1,
    9, 8, 1, 1, 1;

  DataColumn<int> groups(30);
  groups <<
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2;

  Tree<long double, int> result = train_glda(
    data,
    groups,
    0);

  Tree<long double, int> expect = Tree<long double, int>(
    std::make_unique<Condition<long double, int> >(
      as_projector<long double>({ 0.9753647250984685, -0.19102490285203763, -0.02603961769477166, 0.06033431306913992, -0.08862758318234709 }),
      4.0505145097205055,
      std::make_unique<Condition<long double, int> >(
        as_projector<long double>({ 0.15075268856227853, 0.9830270463921728, -0.013280681282024458, 0.023289310653985006, 0.10105782733996031 }),
        2.8568896254203113,
        std::make_unique<Response<long double, int> >(0),
        std::make_unique<Response<long double, int> >(1)),
      std::make_unique<Response<long double, int> >(2)),
    TrainingSpec<long double, int>::lda());

  ASSERT_EQ(expect, result);
}

TEST(PPTreeTrainPDA, lambda_onehalf_univariate_two_groups) {
  Data<long double> data(10, 1);
  data <<
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2;

  DataColumn<int> groups(10, 1);
  groups <<
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1;

  Tree<long double, int> result = pptree::train_glda(
    data,
    groups,
    0.5);

  Tree<long double, int> expect = Tree<long double, int>(
    std::make_unique<Condition<long double, int> >(
      as_projector<long double>({ 1.0 }),
      1.5,
      std::make_unique<Response<long double, int> >(0),
      std::make_unique<Response<long double, int> >(1)),
    TrainingSpec<long double, int>::lda());

  ASSERT_EQ(expect, result);
}

TEST(PPTreeTrainPDA, lambda_onehalf_multivariate_two_groups) {
  Data<long double> data(10, 12);
  data <<
    1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    4, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2,
    5, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 0, 0, 3, 2, 2, 2, 2, 2, 2, 2, 2,
    4, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2,
    4, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
    4, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2;

  DataColumn<int> groups(10);
  groups <<
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1;

  Tree<long double, int> result = pptree::train_glda(
    data,
    groups,
    0.5);

  Tree<long double, int> expect = Tree<long double, int>(
    std::make_unique<Condition<long double, int> >(
      as_projector<long double>({ 0.9969498534721803, -0.00784130658079787, 0.053487283057874875, -0.05254780467349118, -0.007135670500966689, -0.007135670500966691, -0.007135670500966693, -0.007135670500966691, -0.007135670500966698, -0.007135670500966698, -0.007135670500966696, -0.007135670500966696 }),
      2.4440,
      std::make_unique<Response<long double, int> >(0),
      std::make_unique<Response<long double, int> >(1)
      ),
    TrainingSpec<long double, int>::lda()
    );

  ASSERT_EQ(expect, result);
}

TEST(PPTreeTrainForestLDA, all_variables_multivariate_three_groups) {
  Data<long double> data(30, 5);
  data <<
    1, 0, 1, 1, 1,
    1, 0, 1, 0, 0,
    1, 0, 0, 0, 1,
    1, 0, 1, 2, 1,
    1, 0, 0, 1, 1,
    1, 1, 1, 1, 0,
    1, 0, 0, 2, 1,
    1, 0, 1, 1, 2,
    1, 0, 0, 2, 0,
    1, 0, 2, 1, 0,
    2, 5, 0, 0, 1,
    2, 5, 0, 0, 2,
    3, 5, 1, 0, 2,
    2, 5, 1, 0, 1,
    2, 5, 0, 1, 1,
    2, 5, 0, 1, 2,
    2, 5, 2, 1, 1,
    2, 5, 1, 1, 1,
    2, 5, 1, 1, 2,
    2, 5, 2, 1, 2,
    2, 5, 1, 2, 1,
    2, 5, 2, 1, 1,
    9, 8, 0, 0, 1,
    9, 8, 0, 0, 2,
    9, 8, 1, 0, 2,
    9, 8, 1, 0, 1,
    9, 8, 0, 1, 1,
    9, 8, 0, 1, 2,
    9, 8, 2, 1, 1,
    9, 8, 1, 1, 1;

  DataColumn<int> groups(30);
  groups <<
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2;

  const int n_vars = data.cols();
  const double lambda = 0;
  const double seed = 0;


  Forest<long double, int> result = pptree::train_forest_glda(
    data,
    groups,
    4,
    n_vars,
    lambda,
    seed);



  Forest<long double, int> expect(TrainingSpec<long double, int>::uniform_glda(n_vars, lambda, seed));

  expect.add_tree(
    std::make_unique<Tree<long double, int> >(
      Tree<long double, int>(
        std::make_unique<Condition<long double, int> >(
          as_projector<long double>({ 0.3959386339593606, -0.908269881092349, -0.05734470673819268, 0.08786184419030313, -0.0852660670107132 }),
          -1.7110096455357062,
          std::make_unique<Condition<long double, int> >(
            as_projector<long double>({ -1.803617104793913e-15, 1.0, -0.0, 0.0, 0.0 }),
            6.49999999999999,
            std::make_unique<Response<long double, int> >(1),
            std::make_unique<Response<long double, int> >(2)
            ),
          std::make_unique<Response<long double, int> >(0)
          ),
        TrainingSpec<long double, int>::uniform_glda(n_vars, lambda, seed)
        )
      )
    );

  expect.add_tree(
    std::make_unique<Tree<long double, int> >(
      Tree<long double, int>(
        std::make_unique<Condition<long double, int> >(
          as_projector<long double>({ 0.9429063113601566, -0.1791114759384774, -0.1577467223900484, 0.07803343725612998, -0.21880018608199273 }),
          3.745534637999798,
          std::make_unique<Condition<long double, int> >(
            as_projector<long double>({ 0.09705833399783349, 0.9801862846848319, -0.027094022564766305, -0.005144434664196154, 0.17045226854036435 }),
            2.8344748952402568,
            std::make_unique<Response<long double, int> >(0),
            std::make_unique<Response<long double, int> >(1)
            ),
          std::make_unique<Response<long double, int> >(2)
          ),
        TrainingSpec<long double, int>::uniform_glda(n_vars, lambda, seed)
        )
      )
    );


  expect.add_tree(
    std::make_unique<Tree<long double, int> >(
      Tree<long double, int>(
        std::make_unique<Condition<long double, int> >(
          as_projector<long double>({ 0.9541878806121409, -0.1693811442451475, -0.11681578122561698, 0.07761527693154766, -0.20289272660843075 }),
          3.8815126994794134,
          std::make_unique<Condition<long double, int> >(
            as_projector<long double>({ 0.07964113289220642, 0.9872406081485949, -0.019516630829984314, 0.04771950247256534, 0.1278875356665199 }),
            2.8098223049798854,
            std::make_unique<Response<long double, int> >(0),
            std::make_unique<Response<long double, int> >(1)
            ),
          std::make_unique<Response<long double, int> >(2)
          ),
        TrainingSpec<long double, int>::uniform_glda(n_vars, lambda, seed)
        )
      )
    );


  expect.add_tree(
    std::make_unique<Tree<long double, int> >(
      Tree<long double, int>(
        std::make_unique<Condition<long double, int> >(
          as_projector<long double>({ 0.9774257025571, -0.20095147055468665, 0.012604451134322587, -0.015914730117438724, -0.06201089936099892 }),
          3.9924452660765164,
          std::make_unique<Condition<long double, int> >(
            as_projector<long double>({ 1.0, 2.0286166352810033e-15, -0.0, -0.0, -0.0 }),
            1.5000000000000053,
            std::make_unique<Response<long double, int> >(0),
            std::make_unique<Response<long double, int> >(1)
            ),
          std::make_unique<Response<long double, int> >(2)
          ),
        TrainingSpec<long double, int>::uniform_glda(n_vars, lambda, seed)
        )
      )
    );



  ASSERT_EQ(expect, result);
}

TEST(PPTreeTrainForestLDA, some_variables_multivariate_three_groups) {
  Data<long double> data(30, 5);
  data <<
    1, 0, 1, 1, 1,
    1, 0, 1, 0, 0,
    1, 0, 0, 0, 1,
    1, 0, 1, 2, 1,
    1, 0, 0, 1, 1,
    1, 1, 1, 1, 0,
    1, 0, 0, 2, 1,
    1, 0, 1, 1, 2,
    1, 0, 0, 2, 0,
    1, 0, 2, 1, 0,
    2, 5, 0, 0, 1,
    2, 5, 0, 0, 2,
    3, 5, 1, 0, 2,
    2, 5, 1, 0, 1,
    2, 5, 0, 1, 1,
    2, 5, 0, 1, 2,
    2, 5, 2, 1, 1,
    2, 5, 1, 1, 1,
    2, 5, 1, 1, 2,
    2, 5, 2, 1, 2,
    2, 5, 1, 2, 1,
    2, 5, 2, 1, 1,
    9, 8, 0, 0, 1,
    9, 8, 0, 0, 2,
    9, 8, 1, 0, 2,
    9, 8, 1, 0, 1,
    9, 8, 0, 1, 1,
    9, 8, 0, 1, 2,
    9, 8, 2, 1, 1,
    9, 8, 1, 1, 1;

  DataColumn<int> groups(30);
  groups <<
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2;


  const int n_vars = 2;
  const double lambda = 0;
  const double seed = 1;


  Forest<long double, int> result = pptree::train_forest_glda(
    data,
    groups,
    4,
    n_vars,
    lambda,
    seed);


  Forest<long double, int> expect(TrainingSpec<long double, int>::uniform_glda(n_vars, lambda, seed));

  expect.add_tree(
    std::make_unique<Tree<long double, int> >(
      Tree<long double, int>(
        std::make_unique<Condition<long double, int> >(
          as_projector<long double>({ 0.0, 0.9993353191880817, 0.0, -0.03645435259684161, 0.0 }),
          3.116920055735142,
          std::make_unique<Response<long double, int> >(0),
          std::make_unique<Condition<long double, int> >(
            as_projector<long double>({ 0.0, 0.0, 0.908629785316463, 0.4176025780999904, 0.0 }),
            1.1010826217897414,
            std::make_unique<Response<long double, int> >(2),
            std::make_unique<Response<long double, int> >(1)
            )
          ),
        TrainingSpec<long double, int>::uniform_glda(n_vars, lambda, seed)
        )
      )
    );

  expect.add_tree(
    std::make_unique<Tree<long double, int> >(
      Tree<long double, int>(
        std::make_unique<Condition<long double, int> >(
          as_projector<long double>({ 0.0, 0.0, 0.0, 1.0, 0.0 }),
          0.9750000000000001,
          std::make_unique<Condition<long double, int> >(
            as_projector<long double>({ 0.0, 1.0, 0.0, 0.0, 0.0 }),
            6.5,
            std::make_unique<Response<long double, int> >(1),
            std::make_unique<Response<long double, int> >(2)
            ),
          std::make_unique<Response<long double, int> >(0)
          ),
        TrainingSpec<long double, int>::uniform_glda(n_vars, lambda, seed)
        )
      )
    );

  expect.add_tree(
    std::make_unique<Tree<long double, int> >(
      Tree<long double, int>(
        std::make_unique<Condition<long double, int> >(
          as_projector<long double>({ 0.0, 0.0, 1.0, 0.0, 0.0 }),
          0.9305555555555556,
          std::make_unique<Condition<long double, int> >(
            as_projector<long double>({ 0.0, 0.0, 0.0, 1.0, 0.0 }),
            0.8125,
            std::make_unique<Response<long double, int> >(2),
            std::make_unique<Response<long double, int> >(0)
            ),
          std::make_unique<Response<long double, int> >(1)
          ),
        TrainingSpec<long double, int>::uniform_glda(n_vars, lambda, seed)
        )
      )
    );

  expect.add_tree(
    std::make_unique<Tree<long double, int> >(
      Tree<long double, int>(
        std::make_unique<Condition<long double, int> >(
          as_projector<long double>({ 0.0, 0.0, 0.0, 0.7934710960375211, -0.6086079359924704 }),
          0.17291907924993222,
          std::make_unique<Condition<long double, int> >(
            as_projector<long double>({ 0.0, 0.0, 0.0, 1.0, 0.0 }),
            0.6666666666666667,
            std::make_unique<Response<long double, int> >(2),
            std::make_unique<Response<long double, int> >(1)
            ),
          std::make_unique<Response<long double, int> >(0)
          ),
        TrainingSpec<long double, int>::uniform_glda(n_vars, lambda, seed)
        )
      )
    );


  ASSERT_EQ(expect, result);
}


TEST(PPTreeTrainForestPDA, all_variables_multivariate_two_groups) {
  Data<long double> data(10, 12);
  data <<
    1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    4, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2,
    5, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 0, 0, 3, 2, 2, 2, 2, 2, 2, 2, 2,
    4, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2,
    4, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
    4, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2;

  DataColumn<int> groups(10);
  groups <<
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1;

  const int n_vars = data.cols();
  const double lambda = 0.1;
  const double seed = 0;

  Forest<long double, int> result = pptree::train_forest_glda(
    data,
    groups,
    4,
    n_vars,
    lambda,
    seed);


  Forest<long double, int> expect(TrainingSpec<long double, int>::uniform_glda(n_vars, lambda, seed));

  expect.add_tree(
    std::make_unique<Tree<long double, int> >(
      Tree<long double, int>(
        std::make_unique<Condition<long double, int> >(
          as_projector<long double>({ 0.9655843123155974, 0.08681796476306242, 0.09953066811975722, -0.1290329926431043, -0.06476274943747946, -0.0647627494374795, -0.06476274943747948, -0.06476274943747948, -0.06476274943747941, -0.06476274943747942, -0.06476274943747944, -0.06476274943747942 }),
          1.66054084868256,
          std::make_unique<Response<long double, int> >(0),
          std::make_unique<Response<long double, int> >(1)
          ),
        TrainingSpec<long double, int>::uniform_glda(n_vars, lambda, seed)
        )
      )
    );

  expect.add_tree(
    std::make_unique<Tree<long double, int> >(
      Tree<long double, int>(
        std::make_unique<Condition<long double, int> >(
          as_projector<long double>({ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }),
          2.5,
          std::make_unique<Response<long double, int> >(0),
          std::make_unique<Response<long double, int> >(1)
          ),
        TrainingSpec<long double, int>::uniform_glda(n_vars, lambda, seed)
        )
      )
    );

  expect.add_tree(
    std::make_unique<Tree<long double, int> >(
      Tree<long double, int>(
        std::make_unique<Condition<long double, int> >(
          as_projector<long double>({ 0.9607071714826417, 0.13071424952758612, 0.18649074813826336, -0.06675108245672948, -0.05089328231941315, -0.05089328231941322, -0.05089328231941318, -0.05089328231941322, -0.05089328231941308, -0.050893282319413084, -0.05089328231941308, -0.0508932823194131 }),
          1.9601144528047953,
          std::make_unique<Response<long double, int> >(0),
          std::make_unique<Response<long double, int> >(1)
          ),
        TrainingSpec<long double, int>::uniform_glda(n_vars, lambda, seed)
        )
      )
    );

  expect.add_tree(
    std::make_unique<Tree<long double, int> >(
      Tree<long double, int>(
        std::make_unique<Condition<long double, int> >(
          as_projector<long double>({ 0.9611377266908191, 0.027169137836983416, 0.08147067425776834, -0.05364933895611067, -0.09080224800793527, -0.0908022480079353, -0.0908022480079353, -0.09080224800793528, -0.09080224800793517, -0.09080224800793522, -0.09080224800793522, -0.09080224800793524 }),
          1.3347158081570496,
          std::make_unique<Response<long double, int> >(0),
          std::make_unique<Response<long double, int> >(1)
          ),
        TrainingSpec<long double, int>::uniform_glda(n_vars, lambda, seed)
        )
      )
    );

  ASSERT_EQ(expect, result);
}

TEST(PPTreeForestPredictDataColumn, some_variables_multivariate_three_groups) {
  const int n_vars = 2;
  const double lambda = 0;
  const double seed = 1;

  Forest<long double, int> forest(TrainingSpec<long double, int>::uniform_glda(n_vars, lambda, seed));

  forest.add_tree(
    std::make_unique<Tree<long double, int> >(
      Tree<long double, int>(
        std::make_unique<Condition<long double, int> >(
          as_projector<long double>({ 0.0, 0.0, 0.0, 0.5982325379690726, -0.8013225508589422 }),
          -0.3483987096124312,
          std::make_unique<Condition<long double, int> >(
            as_projector<long double>({ 0.999534397402818, 0.0, -0.030512102657559676, 0.0, 0.0 }),
            5.55339996020167,
            std::make_unique<Response<long double, int> >(1),
            std::make_unique<Response<long double, int> >(2)
            ),
          std::make_unique<Response<long double, int> >(0)
          ),
        TrainingSpec<long double, int>::uniform_glda(n_vars, lambda, seed)
        )
      )
    );


  forest.add_tree(
    std::make_unique<Tree<long double, int> >(
      Tree<long double, int>(
        std::make_unique<Condition<long double, int> >(
          as_projector<long double>({ 0.9998222455113714, 0.0, -0.018854107791118468, 0.0, 0.0 }),
          5.300417766337716,
          std::make_unique<Condition<long double, int> >(
            as_projector<long double>({ 0.9989543519613864, 0.0, 0.0, 0.0457187346435417, 0.0 }),
            1.6094899541803496,
            std::make_unique<Response<long double, int> >(0),
            std::make_unique<Response<long double, int> >(1)
            ),
          std::make_unique<Response<long double, int> >(2)
          ),
        TrainingSpec<long double, int>::uniform_glda(n_vars, lambda, seed)
        )
      )
    );


  forest.add_tree(
    std::make_unique<Tree<long double, int> >(
      Tree<long double, int>(
        std::make_unique<Condition<long double, int> >(
          as_projector<long double>({ 0.9741975531020036, -0.2256969816591904, 0.0, 0.0, 0.0 }),
          3.9550147456664178,
          std::make_unique<Condition<long double, int> >(
            as_projector<long double>({ 0.0, 0.9995561292785718, -0.029791683766431428, 0.0, 0.0 }),
            2.6217629631670403,
            std::make_unique<Response<long double, int> >(0),
            std::make_unique<Response<long double, int> >(1)
            ),
          std::make_unique<Response<long double, int> >(2)
          ),
        TrainingSpec<long double, int>::uniform_glda(n_vars, lambda, seed)
        )
      )
    );


  forest.add_tree(
    std::make_unique<Tree<long double, int> >(
      Tree<long double, int>(
        std::make_unique<Condition<long double, int> >(
          as_projector<long double>({ 0.9615748657636985, 0.0, 0.0, 0.0, -0.2745428519038971 }),
          4.734758305714628,
          std::make_unique<Condition<long double, int> >(
            as_projector<long double>({ 0.0, 0.0, 0.3772334858435029, 0.0, -0.926118187467647 }),
            -0.8315603229605784,
            std::make_unique<Response<long double, int> >(1),
            std::make_unique<Response<long double, int> >(0)
            ),
          std::make_unique<Response<long double, int> >(2)
          ),
        TrainingSpec<long double, int>::uniform_glda(n_vars, lambda, seed)
        )
      )
    );

  DataColumn<long double> data(5);
  data << 9, 8, 1, 1, 1;

  int result = forest.predict(data);

  ASSERT_EQ(2, result);
}

TEST(PPTreeForestPredictData, some_variables_multivariate_three_groups) {
  const int n_vars = 2;
  const double lambda = 0;
  const double seed = 1;

  Forest<long double, int> forest(TrainingSpec<long double, int>::uniform_glda(n_vars, lambda, seed));

  forest.add_tree(
    std::make_unique<Tree<long double, int> >(
      Tree<long double, int>(
        std::make_unique<Condition<long double, int> >(
          as_projector<long double>({ 0.0, 0.0, 0.0, 0.5982325379690726, -0.8013225508589422 }),
          -0.3483987096124312,
          std::make_unique<Condition<long double, int> >(
            as_projector<long double>({ 0.999534397402818, 0.0, -0.030512102657559676, 0.0, 0.0 }),
            5.55339996020167,
            std::make_unique<Response<long double, int> >(1),
            std::make_unique<Response<long double, int> >(2)
            ),
          std::make_unique<Response<long double, int> >(0)
          ),
        TrainingSpec<long double, int>::uniform_glda(n_vars, lambda, seed)
        )
      )
    );


  forest.add_tree(
    std::make_unique<Tree<long double, int> >(
      Tree<long double, int>(
        std::make_unique<Condition<long double, int> >(
          as_projector<long double>({ 0.9998222455113714, 0.0, -0.018854107791118468, 0.0, 0.0 }),
          5.300417766337716,
          std::make_unique<Condition<long double, int> >(
            as_projector<long double>({ 0.9989543519613864, 0.0, 0.0, 0.0457187346435417, 0.0 }),
            1.6094899541803496,
            std::make_unique<Response<long double, int> >(0),
            std::make_unique<Response<long double, int> >(1)
            ),
          std::make_unique<Response<long double, int> >(2)
          ),
        TrainingSpec<long double, int>::uniform_glda(n_vars, lambda, seed)
        )
      )
    );


  forest.add_tree(
    std::make_unique<Tree<long double, int> >(
      Tree<long double, int>(
        std::make_unique<Condition<long double, int> >(
          as_projector<long double>({ 0.9741975531020036, -0.2256969816591904, 0.0, 0.0, 0.0 }),
          3.9550147456664178,
          std::make_unique<Condition<long double, int> >(
            as_projector<long double>({ 0.0, 0.9995561292785718, -0.029791683766431428, 0.0, 0.0 }),
            2.6217629631670403,
            std::make_unique<Response<long double, int> >(0),
            std::make_unique<Response<long double, int> >(1)
            ),
          std::make_unique<Response<long double, int> >(2)
          ),
        TrainingSpec<long double, int>::uniform_glda(n_vars, lambda, seed)
        )
      )
    );


  forest.add_tree(
    std::make_unique<Tree<long double, int> >(
      Tree<long double, int>(
        std::make_unique<Condition<long double, int> >(
          as_projector<long double>({ 0.9615748657636985, 0.0, 0.0, 0.0, -0.2745428519038971 }),
          4.734758305714628,
          std::make_unique<Condition<long double, int> >(
            as_projector<long double>({ 0.0, 0.0, 0.3772334858435029, 0.0, -0.926118187467647 }),
            -0.8315603229605784,
            std::make_unique<Response<long double, int> >(1),
            std::make_unique<Response<long double, int> >(0)
            ),
          std::make_unique<Response<long double, int> >(2)
          ),
        TrainingSpec<long double, int>::uniform_glda(n_vars, lambda, seed)
        )
      )
    );

  Data<long double> data(30, 5);
  data <<
    1, 0, 1, 1, 1,
    1, 0, 1, 0, 0,
    1, 0, 0, 0, 1,
    1, 0, 1, 2, 1,
    1, 0, 0, 1, 1,
    1, 1, 1, 1, 0,
    1, 0, 0, 2, 1,
    1, 0, 1, 1, 2,
    1, 0, 0, 2, 0,
    1, 0, 2, 1, 0,
    2, 5, 0, 0, 1,
    2, 5, 0, 0, 2,
    3, 5, 1, 0, 2,
    2, 5, 1, 0, 1,
    2, 5, 0, 1, 1,
    2, 5, 0, 1, 2,
    2, 5, 2, 1, 1,
    2, 5, 1, 1, 1,
    2, 5, 1, 1, 2,
    2, 5, 2, 1, 2,
    2, 5, 1, 2, 1,
    2, 5, 2, 1, 1,
    9, 8, 0, 0, 1,
    9, 8, 0, 0, 2,
    9, 8, 1, 0, 2,
    9, 8, 1, 0, 1,
    9, 8, 0, 1, 1,
    9, 8, 0, 1, 2,
    9, 8, 2, 1, 1,
    9, 8, 1, 1, 1;

  DataColumn<int> groups(30);
  groups <<
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    1,
    1,
    0,
    0,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2;

  DataColumn<int> result = forest.predict(data);

  ASSERT_EQ(groups.size(), result.size());
  ASSERT_EQ(groups.cols(), result.cols());
  ASSERT_EQ(groups.rows(), result.rows());
  ASSERT_EQ(groups, result);
}


TEST(PPTreePredictDataColumn, univariate_two_groups) {
  Tree<long double, int> tree = Tree<long double, int>(
    std::make_unique<Condition<long double, int> >(
      as_projector<long double>({ 1.0 }),
      1.5,
      std::make_unique<Response<long double, int> >(0),
      std::make_unique<Response<long double, int> >(1)),
    TrainingSpec<long double, int>::lda());


  DataColumn<long double> input(1);
  input << 1.0;
  ASSERT_EQ(tree.predict(input), 0);

  input << 2.0;
  ASSERT_EQ(tree.predict(input), 1);
}

TEST(PPTreePredictDataColumn, univariate_three_groups) {
  Tree<long double, int> tree = Tree<long double, int>(
    std::make_unique<Condition<long double, int> >(
      as_projector<long double>({ 1.0 }),
      1.75,
      std::make_unique<Response<long double, int> >(0),
      std::make_unique<Condition<long double, int> >(
        as_projector<long double>({ 1.0 }),
        2.5,
        std::make_unique<Response<long double, int> >(1),
        std::make_unique<Response<long double, int> >(2))),
    TrainingSpec<long double, int>::lda());

  DataColumn<long double> input(1);
  input << 1.0;
  ASSERT_EQ(tree.predict(input), 0);

  input << 2.0;
  ASSERT_EQ(tree.predict(input), 1);

  input << 3.0;
  ASSERT_EQ(tree.predict(input), 2);
}

TEST(PPTreePredictDataColumn, multivariate_two_groups) {
  Tree<long double, int> tree = Tree<long double, int>(
    std::make_unique<Condition<long double, int> >(
      as_projector<long double>({ 1.0, 0.0, 0.0, 0.0 }),
      2.5,
      std::make_unique<Response<long double, int> >(0),
      std::make_unique<Response<long double, int> >(1)),
    TrainingSpec<long double, int>::lda());

  DataColumn<long double> input(4);
  input << 1, 0, 1, 1;
  ASSERT_EQ(tree.predict(input), 0);

  input << 4, 0, 0, 1;
  ASSERT_EQ(tree.predict(input), 1);
}

TEST(PPTreePredictDataColumn, multivariate_three_groups) {
  Tree<long double, int> tree = Tree<long double, int>(
    std::make_unique<Condition<long double, int> >(
      as_projector<long double>({ 0.9805806756909201, -0.19611613513818427, 0.0, 0.0, 0.0 }),
      4.118438837901864,
      std::make_unique<Condition<long double, int> >(
        as_projector<long double>({ 0.0, 1.0, 0.0, 0.0, 0.0 }),
        2.5,
        std::make_unique<Response<long double, int> >(0),
        std::make_unique<Response<long double, int> >(1)),
      std::make_unique<Response<long double, int> >(2)),
    TrainingSpec<long double, int>::lda());

  DataColumn<long double> input(5);
  input << 1, 0, 0, 1, 1;
  ASSERT_EQ(tree.predict(input), 0);

  input << 2, 5, 0, 0, 1;
  ASSERT_EQ(tree.predict(input), 1);

  input << 9, 8, 0, 0, 1;
  ASSERT_EQ(tree.predict(input), 2);
}

TEST(PPTreePredictData, univariate_two_groups) {
  Tree<long double, int> tree = Tree<long double, int>(
    std::make_unique<Condition<long double, int> >(
      as_projector<long double>({ 1.0 }),
      1.5,
      std::make_unique<Response<long double, int> >(0),
      std::make_unique<Response<long double, int> >(1)),
    TrainingSpec<long double, int>::lda());

  Data<long double> input(2, 1);
  input << 1.0,  2.0;

  DataColumn<int> result = tree.predict(input);

  DataColumn<int> expected(2);
  expected << 0, 1;

  ASSERT_EQ(result, expected);
}

TEST(PPTreePredictData, univariate_three_groups) {
  Tree<long double, int> tree = Tree<long double, int>(
    std::make_unique<Condition<long double, int> >(
      as_projector<long double>({ 1.0 }),
      1.75,
      std::make_unique<Response<long double, int> >(0),
      std::make_unique<Condition<long double, int> >(
        as_projector<long double>({ 1.0 }),
        2.5,
        std::make_unique<Response<long double, int> >(1),
        std::make_unique<Response<long double, int> >(2))),
    TrainingSpec<long double, int>::lda());

  Data<long double> input(3, 1);
  input << 1.0, 2.0, 3.0;

  DataColumn<int> result = tree.predict(input);

  DataColumn<int> expected(3);
  expected << 0, 1, 2;

  ASSERT_EQ(result, expected);
}

TEST(PPTreePredictData, multivariate_two_groups) {
  Tree<long double, int> tree = Tree<long double, int>(
    std::make_unique<Condition<long double, int> >(
      as_projector<long double>({ 1.0, 0.0, 0.0, 0.0 }),
      2.5,
      std::make_unique<Response<long double, int> >(0),
      std::make_unique<Response<long double, int> >(1)),
    TrainingSpec<long double, int>::lda());

  Data<long double> input(2, 4);
  input <<
    1, 0, 1, 1,
    4, 0, 0, 1;

  DataColumn<int> result = tree.predict(input);

  DataColumn<int> expected(2);
  expected << 0, 1;

  ASSERT_EQ(result, expected);
}

TEST(PPTreePredictData, multivariate_three_groups) {
  Tree<long double, int> tree = Tree<long double, int>(
    std::make_unique<Condition<long double, int> >(
      as_projector<long double>({ 0.9805806756909201, -0.19611613513818427, 0.0, 0.0, 0.0 }),
      4.118438837901864,
      std::make_unique<Condition<long double, int> >(
        as_projector<long double>({ 0.0, 1.0, 0.0, 0.0, 0.0 }),
        2.5,
        std::make_unique<Response<long double, int> >(0),
        std::make_unique<Response<long double, int> >(1)),
      std::make_unique<Response<long double, int> >(2)),
    TrainingSpec<long double, int>::lda());

  Data<long double> input(3, 5);
  input <<
    1, 0, 0, 1, 1,
    2, 5, 0, 0, 1,
    9, 8, 0, 0, 1;

  DataColumn<int> result = tree.predict(input);

  DataColumn<int> expected(3);
  expected << 0, 1, 2;

  ASSERT_EQ(result, expected);
}
