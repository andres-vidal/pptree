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
        std::make_unique<Response<long double, int> >(2))));

  Tree<long double, int> t2(
    std::make_unique<Condition<long double, int> >(
      as_projector<long double>({ 1.0, 2.0 }),
      3.0,
      std::make_unique<Response<long double, int> >(1),
      std::make_unique<Condition<long double, int> >(
        as_projector<long double>({ 1.0, 2.0 }),
        3.0,
        std::make_unique<Response<long double, int> >(1),
        std::make_unique<Response<long double, int> >(2))));

  ASSERT_TRUE(t1 == t2);
}

TEST(TreeEquals, false_case) {
  Tree<long double, int> t1(
    std::make_unique<Condition<long double, int> >(
      as_projector<long double>({ 1.0, 2.0 }),
      3.0,
      std::make_unique<Response<long double, int> >(1),
      std::make_unique<Response<long double, int> >(2)));

  Tree<long double, int> t2(
    std::make_unique<Condition<long double, int> >(
      as_projector<long double>({ 1.0, 2.0 }),
      3.0,
      std::make_unique<Response<long double, int> >(1),
      std::make_unique<Response<long double, int> >(3)));

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
      std::make_unique<Response<long double, int> >(1)));

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
        std::make_unique<Response<long double, int> >(2))));


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
      )
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
      std::make_unique<Response<long double, int> >(2)));

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
      std::make_unique<Response<long double, int> >(1)));

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
      )
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

  std::mt19937 generator(1);

  Forest<long double, int> result = pptree::train_forest_glda(
    data,
    groups,
    4,
    data.cols(),
    0,
    generator);

  Forest<long double, int> expect;

  expect.add_tree(
    std::make_unique<Tree<long double, int> >(
      Tree<long double, int>(
        std::make_unique<Condition<long double, int> >(
          as_projector<long double>({ 0.9631149682241399, -0.18918627608455488, -0.05737102823648358, 0.07944045935537511, -0.16436511022832137 }),
          3.886824734352277,
          std::make_unique<Condition<long double, int> >(
            as_projector<long double>({ 0.05662843789392038, 0.9390689214273856, -0.04305153782509235, 0.12364262413109396, 0.31273286910680315 }),
            3.0140930896250215,
            std::make_unique<Response<long double, int> >(0),
            std::make_unique<Response<long double, int> >(1)
            ),
          std::make_unique<Response<long double, int> >(2)
          )
        )
      )
    );

  expect.add_tree(
    std::make_unique<Tree<long double, int> >(
      Tree<long double, int>(
        std::make_unique<Condition<long double, int> >(
          as_projector<long double>({ 0.9603485139189338, -0.17496307094186994, -0.04225601427843259, 0.1507998001134798, -0.15030803426071773 }),
          4.0485685231690995,
          std::make_unique<Condition<long double, int> >(
            as_projector<long double>({ 0.09569190005776607, 0.9798250377436776, -0.010165041862385826, -0.05752095361036075, 0.1654508008249762 }),
            2.8367314322123316,
            std::make_unique<Response<long double, int> >(0),
            std::make_unique<Response<long double, int> >(1)
            ),
          std::make_unique<Response<long double, int> >(2)
          )
        )
      )
    );

  expect.add_tree(
    std::make_unique<Tree<long double, int> >(
      Tree<long double, int>(
        std::make_unique<Condition<long double, int> >(
          as_projector<long double>({ 0.9718365611401335, -0.19549983612592006, -0.041085639915505054, 0.060726386865359874, -0.10926018779532659 }),
          3.9642503461040244,
          std::make_unique<Condition<long double, int> >(
            as_projector<long double>({ 0.2092935388855266, 0.9560642461190949, -0.019346999921465723, 0.06730876793869753, 0.19295749590547875 }),
            3.0587113947307247,
            std::make_unique<Response<long double, int> >(0),
            std::make_unique<Response<long double, int> >(1)
            ),
          std::make_unique<Response<long double, int> >(2)
          )
        )
      )
    );

  expect.add_tree(
    std::make_unique<Tree<long double, int> >(
      Tree<long double, int>(
        std::make_unique<Condition<long double, int> >(
          as_projector<long double>({ 0.9752790469829162, -0.1928709522859401, -0.03773775720254776, 0.05783794961167642, -0.08283845451508612 }),
          4.016492286925132,
          std::make_unique<Condition<long double, int> >(
            as_projector<long double>({ 0.2749387669196902, 0.8939606152049122, 0.040183988430902706, 0.19516474299595846, 0.29247061916069805 }),
            3.316983433613052,
            std::make_unique<Response<long double, int> >(0),
            std::make_unique<Response<long double, int> >(1)
            ),
          std::make_unique<Response<long double, int> >(2)
          )
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

  std::mt19937 generator(1);

  Forest<long double, int> result = pptree::train_forest_glda(
    data,
    groups,
    4,
    2,
    0,
    generator);

  Forest<long double, int> expect;

  expect.add_tree(
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
          )
        )
      )
    );


  expect.add_tree(
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
          )
        )
      )
    );


  expect.add_tree(
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
          )
        )
      )
    );


  expect.add_tree(
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
          )
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

  std::mt19937 generator(0);

  Forest<long double, int> result = pptree::train_forest_glda(
    data,
    groups,
    4,
    data.cols(),
    0.1,
    generator);

  Forest<long double, int> expect;

  expect.add_tree(
    std::make_unique<Tree<long double, int> >(
      Tree<long double, int>(
        std::make_unique<Condition<long double, int> >(
          as_projector<long double>({ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }),
          2.5,
          std::make_unique<Response<long double, int> >(0),
          std::make_unique<Response<long double, int> >(1)
          )
        )
      )
    );

  expect.add_tree(
    std::make_unique<Tree<long double, int> >(
      Tree<long double, int>(
        std::make_unique<Condition<long double, int> >(
          as_projector<long double>({ 0.930054052413662, 0.045822537426335075, 0.2513198292993562, -0.24141500684671008, -0.03784324993126796, -0.03784324993126797, -0.03784324993126797, -0.03784324993126796, -0.03784324993126759, -0.037843249931267586, -0.037843249931267586, -0.03784324993126759 }),
          1.8089387215062656,
          std::make_unique<Response<long double, int> >(0),
          std::make_unique<Response<long double, int> >(1)
          )
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
          )
        )
      )
    );

  expect.add_tree(
    std::make_unique<Tree<long double, int> >(
      Tree<long double, int>(
        std::make_unique<Condition<long double, int> >(
          as_projector<long double>({ 0.941548245944998, 0.13544597103198155, 0.06283022647715046, 0.060289643232506254, -0.10461764373710465, -0.10461764373710344, -0.10461764373710347, -0.10461764373710342, -0.10461764373710364, -0.10461764373710364, -0.10461764373710364, -0.10461764373710362 }),
          1.2795964320491389,
          std::make_unique<Response<long double, int> >(0),
          std::make_unique<Response<long double, int> >(1)
          )
        )
      )
    );


  ASSERT_EQ(expect, result);
}

TEST(PPTreeForestPredictDataColumn, some_variables_multivariate_three_groups) {
  Forest<long double, int> forest;

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
          )
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
          )
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
          )
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
          )
        )
      )
    );

  DataColumn<long double> data(5);
  data << 9, 8, 1, 1, 1;

  int result = forest.predict(data);

  ASSERT_EQ(2, result);
}

TEST(PPTreeForestPredictData, some_variables_multivariate_three_groups) {
  Forest<long double, int> forest;

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
          )
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
          )
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
          )
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
          )
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
      std::make_unique<Response<long double, int> >(1)));


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
        std::make_unique<Response<long double, int> >(2))));

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
      std::make_unique<Response<long double, int> >(1)));

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
      std::make_unique<Response<long double, int> >(2)));

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
      std::make_unique<Response<long double, int> >(1)));

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
        std::make_unique<Response<long double, int> >(2))));

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
      std::make_unique<Response<long double, int> >(1)));

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
      std::make_unique<Response<long double, int> >(2)));

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
