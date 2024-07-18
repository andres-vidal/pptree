#include <gtest/gtest.h>

#include "Forest.hpp"
#include "VIStrategy.hpp"

using namespace models;
using namespace models::stats;
using namespace models::pp;
using namespace models::math;

static Projector<long double> as_projector(std::vector<long double> vector) {
  Eigen::Map<Projector<long double> > projector(vector.data(), vector.size());
  return projector;
}

TEST(Forest, TrainLDAAllVariablesMultivariateThreeGroups) {
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
  const int seed = 0;


  Forest<long double, int> result = Forest<long double, int>::train(
    *TrainingSpec<long double, int>::uniform_glda(n_vars, lambda),
    DataSpec<long double, int>(data, groups),
    4,
    seed);

  Forest<long double, int> expect(
    TrainingSpec<long double, int>::uniform_glda(n_vars, lambda),
    std::make_shared<DataSpec<long double, int> >(data, groups, std::set<int>({ 0, 1, 2 })),
    seed);

  expect.add_tree(
    std::make_unique<BootstrapTree<long double, int> >(
      std::make_unique<Condition<long double, int> >(
        as_projector({ 0.3959386339593606, -0.908269881092349, -0.05734470673819268, 0.08786184419030313, -0.0852660670107132 }),
        -1.7110096455357062,
        std::make_unique<Condition<long double, int> >(
          as_projector({ -1.803617104793913e-15, 1.0, -0.0, 0.0, 0.0 }),
          6.49999999999999,
          std::make_unique<Response<long double, int> >(1),
          std::make_unique<Response<long double, int> >(2)
          ),
        std::make_unique<Response<long double, int> >(0)
        ),
      TrainingSpec<long double, int>::uniform_glda(n_vars, lambda),
      std::make_shared<BootstrapDataSpec<long double, int> >(data, groups, std::set<int>({ 0, 1, 2 }), std::vector<int>({ 0, 1 })))
    );

  expect.add_tree(
    std::make_unique<BootstrapTree<long double, int> >(
      std::make_unique<Condition<long double, int> >(
        as_projector({ 0.9429063113601566, -0.1791114759384774, -0.1577467223900484, 0.07803343725612998, -0.21880018608199273 }),
        3.745534637999798,
        std::make_unique<Condition<long double, int> >(
          as_projector({ 0.09705833399783349, 0.9801862846848319, -0.027094022564766305, -0.005144434664196154, 0.17045226854036435 }),
          2.8344748952402568,
          std::make_unique<Response<long double, int> >(0),
          std::make_unique<Response<long double, int> >(1)
          ),
        std::make_unique<Response<long double, int> >(2)
        ),
      TrainingSpec<long double, int>::uniform_glda(n_vars, lambda),
      std::make_shared<BootstrapDataSpec<long double, int> >(data, groups, std::set<int>({ 0, 1, 2 }), std::vector<int>({ 0, 1 })))
    );


  expect.add_tree(
    std::make_unique<BootstrapTree<long double, int> >(
      std::make_unique<Condition<long double, int> >(
        as_projector({ 0.9541878806121409, -0.1693811442451475, -0.11681578122561698, 0.07761527693154766, -0.20289272660843075 }),
        3.8815126994794134,
        std::make_unique<Condition<long double, int> >(
          as_projector({ 0.07964113289220642, 0.9872406081485949, -0.019516630829984314, 0.04771950247256534, 0.1278875356665199 }),
          2.8098223049798854,
          std::make_unique<Response<long double, int> >(0),
          std::make_unique<Response<long double, int> >(1)
          ),
        std::make_unique<Response<long double, int> >(2)
        ),
      TrainingSpec<long double, int>::uniform_glda(n_vars, lambda),
      std::make_shared<BootstrapDataSpec<long double, int> >(data, groups, std::set<int>({ 0, 1, 2 }), std::vector<int>({ 0, 1 })))

    );


  expect.add_tree(
    std::make_unique<BootstrapTree<long double, int> >(
      std::make_unique<Condition<long double, int> >(
        as_projector({ 0.9774257025571, -0.20095147055468665, 0.012604451134322587, -0.015914730117438724, -0.06201089936099892 }),
        3.9924452660765164,
        std::make_unique<Condition<long double, int> >(
          as_projector({ 1.0, 2.0286166352810033e-15, -0.0, -0.0, -0.0 }),
          1.5000000000000053,
          std::make_unique<Response<long double, int> >(0),
          std::make_unique<Response<long double, int> >(1)
          ),
        std::make_unique<Response<long double, int> >(2)
        ),
      TrainingSpec<long double, int>::uniform_glda(n_vars, lambda),
      std::make_shared<BootstrapDataSpec<long double, int> >(data, groups, std::set<int>({ 0, 1, 2 }), std::vector<int>({ 0, 1 })))

    );



  ASSERT_EQ(expect, result);
  ASSERT_EQ(expect.seed, result.seed);
}

TEST(Forest, TrainLDASomeVariablesMultivariateThreeGroups) {
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
  const int seed = 1;

  Forest<long double, int> result = Forest<long double, int>::train(
    *TrainingSpec<long double, int>::uniform_glda(n_vars, lambda),
    DataSpec<long double, int>(data, groups),
    4,
    seed);

  Forest<long double, int> expect(
    TrainingSpec<long double, int>::uniform_glda(n_vars, lambda),
    std::make_shared<DataSpec<long double, int> >(data, groups, std::set<int>({ 0, 1, 2 })),
    seed);

  expect.add_tree(
    std::make_unique<BootstrapTree<long double, int> >(
      std::make_unique<Condition<long double, int> >(
        as_projector({ 0.0, 0.9993353191880817, 0.0, -0.03645435259684161, 0.0 }),
        3.116920055735142,
        std::make_unique<Response<long double, int> >(0),
        std::make_unique<Condition<long double, int> >(
          as_projector({ 0.0, 0.0, 0.908629785316463, 0.4176025780999904, 0.0 }),
          1.1010826217897414,
          std::make_unique<Response<long double, int> >(2),
          std::make_unique<Response<long double, int> >(1)
          )
        ),
      TrainingSpec<long double, int>::uniform_glda(n_vars, lambda),
      std::make_shared<BootstrapDataSpec<long double, int> >(data, groups, std::set<int>({ 0, 1, 2 }), std::vector<int>({ 0, 1 }))
      )
    );

  expect.add_tree(
    std::make_unique<BootstrapTree<long double, int> >(
      std::make_unique<Condition<long double, int> >(
        as_projector({ 0.0, 0.996837157947326, 0.0, -0.07947125603322207, 0.0 }),
        3.1123944307990516,
        std::make_unique<Response<long double, int> >(0),
        std::make_unique<Condition<long double, int> >(
          as_projector({ 0.0, 0.0, 0.0, 1.0, 0.0 }),
          0.625,
          std::make_unique<Response<long double, int> >(2),
          std::make_unique<Response<long double, int> >(1)
          )
        ),
      TrainingSpec<long double, int>::uniform_glda(n_vars, lambda),
      std::make_shared<BootstrapDataSpec<long double, int> >(data, groups, std::set<int>({ 0, 1, 2 }), std::vector<int>({ 0, 1 }))
      )
    );

  expect.add_tree(
    std::make_unique<BootstrapTree<long double, int> >(
      std::make_unique<Condition<long double, int> >(
        as_projector({ 0.0, 0.9238600027787576, 0.0, 0.38273057790779436, 0.0 }),
        3.2641847933177965,
        std::make_unique<Response<long double, int> >(0),
        std::make_unique<Condition<long double, int> >(
          as_projector({ 1.0, 0.0, 0.0, 0.0, 0.0 }),
          5.5,
          std::make_unique<Response<long double, int> >(1),
          std::make_unique<Response<long double, int> >(2)
          )
        ),
      TrainingSpec<long double, int>::uniform_glda(n_vars, lambda),
      std::make_shared<BootstrapDataSpec<long double, int> >(data, groups, std::set<int>({ 0, 1, 2 }), std::vector<int>({ 0, 1 }))
      )
    );

  expect.add_tree(
    std::make_unique<BootstrapTree<long double, int> >(
      std::make_unique<Condition<long double, int> >(
        as_projector({ 0.0, 0.0, 0.13105273540218915, 0.0, -0.9913753983953826 }),
        -0.9047649582634063,
        std::make_unique<Condition<long double, int> >(
          as_projector({ 0.0, 1.0, 0.0, 0.0, 0.0 }),
          6.5,
          std::make_unique<Response<long double, int> >(1),
          std::make_unique<Response<long double, int> >(2)
          ),
        std::make_unique<Response<long double, int> >(0)
        ),
      TrainingSpec<long double, int>::uniform_glda(n_vars, lambda),
      std::make_shared<BootstrapDataSpec<long double, int> >(data, groups, std::set<int>({ 0, 1, 2 }), std::vector<int>({ 0, 1 }))
      )
    );

  ASSERT_EQ(expect, result);
  ASSERT_EQ(expect.seed, result.seed);
}

TEST(Forest, TrainPDAAllVariablesMultivariateTwoGroups) {
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
  const int seed = 0;

  Forest<long double, int> result = Forest<long double, int>::train(
    *TrainingSpec<long double, int>::uniform_glda(n_vars, lambda),
    DataSpec<long double, int>(data, groups),
    4,
    seed);


  Forest<long double, int> expect(
    TrainingSpec<long double, int>::uniform_glda(n_vars, lambda),
    std::make_shared<DataSpec<long double, int> >(data, groups, std::set<int>({ 0, 1, 2 })),
    seed);

  expect.add_tree(
    std::make_unique<BootstrapTree<long double, int> >(
      std::make_unique<Condition<long double, int> >(
        as_projector({ 0.9655843123155974, 0.08681796476306242, 0.09953066811975722, -0.1290329926431043, -0.06476274943747946, -0.0647627494374795, -0.06476274943747948, -0.06476274943747948, -0.06476274943747941, -0.06476274943747942, -0.06476274943747944, -0.06476274943747942 }),
        1.66054084868256,
        std::make_unique<Response<long double, int> >(0),
        std::make_unique<Response<long double, int> >(1)
        ),
      TrainingSpec<long double, int>::uniform_glda(n_vars, lambda),
      std::make_shared<BootstrapDataSpec<long double, int> >(data, groups, std::set<int>({ 0, 1, 2 }), std::vector<int>({ 0, 1 })))
    );

  expect.add_tree(
    std::make_unique<BootstrapTree<long double, int> >(
      std::make_unique<Condition<long double, int> >(
        as_projector({ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }),
        2.5,
        std::make_unique<Response<long double, int> >(0),
        std::make_unique<Response<long double, int> >(1)),
      TrainingSpec<long double, int>::uniform_glda(n_vars, lambda),
      std::make_shared<BootstrapDataSpec<long double, int> >(data, groups, std::set<int>({ 0, 1, 2 }), std::vector<int>({ 0, 1 })))
    );

  expect.add_tree(
    std::make_unique<BootstrapTree<long double, int> >(
      std::make_unique<Condition<long double, int> >(
        as_projector({ 0.9607071714826417, 0.13071424952758612, 0.18649074813826336, -0.06675108245672948, -0.05089328231941315, -0.05089328231941322, -0.05089328231941318, -0.05089328231941322, -0.05089328231941308, -0.050893282319413084, -0.05089328231941308, -0.0508932823194131 }),
        1.9601144528047953,
        std::make_unique<Response<long double, int> >(0),
        std::make_unique<Response<long double, int> >(1)),
      TrainingSpec<long double, int>::uniform_glda(n_vars, lambda),
      std::make_shared<BootstrapDataSpec<long double, int> >(data, groups, std::set<int>({ 0, 1, 2 }), std::vector<int>({ 0, 1 })))
    );

  expect.add_tree(
    std::make_unique<BootstrapTree<long double, int> >(
      std::make_unique<Condition<long double, int> >(
        as_projector({ 0.9611377266908191, 0.027169137836983416, 0.08147067425776834, -0.05364933895611067, -0.09080224800793527, -0.0908022480079353, -0.0908022480079353, -0.09080224800793528, -0.09080224800793517, -0.09080224800793522, -0.09080224800793522, -0.09080224800793524 }),
        1.3347158081570496,
        std::make_unique<Response<long double, int> >(0),
        std::make_unique<Response<long double, int> >(1)),
      TrainingSpec<long double, int>::uniform_glda(n_vars, lambda),
      std::make_shared<BootstrapDataSpec<long double, int> >(data, groups, std::set<int>({ 0, 1, 2 }), std::vector<int>({ 0, 1 })))
    );

  ASSERT_EQ(expect, result);
  ASSERT_EQ(expect.seed, result.seed);
}

TEST(Forest, TrainLDAConstantData) {
  Data<long double> x(10, 3);
  x <<
    1, 2, 3,
    1, 2, 3,
    1, 2, 3,
    1, 2, 3,
    1, 2, 3,
    1, 2, 3,
    1, 2, 3,
    1, 2, 3,
    1, 2, 3,
    1, 2, 3;

  DataColumn<int> y(10, 1);
  y <<
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1;

  auto spec = TrainingSpec<long double, int>::glda(0.0);
  auto data = DataSpec<long double, int>(x, y);

  ASSERT_THROW((Forest<long double, int>::train(*spec, data, 4, 0)), models::training_error);
}

TEST(Forest, TrainLDANoVarianceBetweenGroups) {
  Data<long double> x(12, 3);
  x <<
    1, 2, 3,
    1, 2, 3,
    1, 2, 3,
    1, 2, 3,
    7, 8, 9,
    7, 8, 9,
    7, 8, 9,
    7, 8, 9,
    7, 8, 9,
    7, 8, 9,
    7, 8, 9,
    7, 8, 9;

  DataColumn<int> y(12);
  y <<
    0, 0, 0, 0,
    1, 1, 1, 1,
    2, 2, 2, 2;

  auto spec = TrainingSpec<long double, int>::glda(0.0);
  auto data = DataSpec<long double, int>(x, y);

  ASSERT_THROW((Forest<long double, int>::train(*spec, data, 4, 0)), models::training_error);
}

TEST(Forest, TrainLDANoVarianceBetweenGroupsInSomeVariablesNoRetries) {
  Data<long double> x(12, 4);
  x <<
    1, 2, 3, 4,
    1, 2, 3, 4,
    1, 2, 3, 4,
    1, 2, 3, 4,
    7, 8, 9, 5,
    7, 8, 9, 5,
    7, 8, 9, 5,
    7, 8, 9, 5,
    7, 7, 9, 5,
    7, 7, 9, 5,
    7, 7, 9, 5,
    7, 7, 9, 5;

  DataColumn<int> y(12);
  y <<
    0, 0, 0, 0,
    1, 1, 1, 1,
    2, 2, 2, 2;

  auto spec = TrainingSpec<long double, int>::uniform_glda(2, 0.0);
  auto data = DataSpec<long double, int>(x, y);

  ASSERT_THROW((Forest<long double, int>::train(*spec, data, 4, 0)), models::training_error);
}

TEST(Forest, TrainLDANoVarianceBetweenGroupsInSomeVariablesNotEnoughRetries) {
  Data<long double> x(12, 4);
  x <<
    1, 2, 3, 4,
    1, 2, 3, 4,
    1, 2, 3, 4,
    1, 2, 3, 4,
    7, 8, 9, 5,
    7, 8, 9, 5,
    7, 8, 9, 5,
    7, 8, 9, 5,
    7, 7, 9, 5,
    7, 7, 9, 5,
    7, 7, 9, 5,
    7, 7, 9, 5;

  DataColumn<int> y(12);
  y <<
    0, 0, 0, 0,
    1, 1, 1, 1,
    2, 2, 2, 2;

  auto spec = TrainingSpec<long double, int>::uniform_glda(2, 0.0, 1);
  auto data = DataSpec<long double, int>(x, y);

  ASSERT_THROW((Forest<long double, int>::train(*spec, data, 4, 0)), models::training_error);
}

TEST(Forest, TrainLDANoVarianceBetweenGroupsInSomeVariablesEnoughRetries) {
  Data<long double> x(12, 4);
  x <<
    1, 2, 3, 4,
    1, 2, 3, 4,
    1, 2, 3, 4,
    1, 2, 3, 4,
    7, 8, 9, 5,
    7, 8, 9, 5,
    7, 8, 9, 5,
    7, 8, 9, 5,
    7, 7, 9, 5,
    7, 7, 9, 5,
    7, 7, 9, 5,
    7, 7, 9, 5;

  DataColumn<int> y(12);
  y <<
    0, 0, 0, 0,
    1, 1, 1, 1,
    2, 2, 2, 2;

  auto spec = TrainingSpec<long double, int>::uniform_glda(2, 0.0, 3);
  auto data = DataSpec<long double, int>(x, y);

  ASSERT_NO_THROW((Forest<long double, int>::train(*spec, data, 4, 0)));
}


TEST(Forest, PredictDataColumnSomeVariablesMultivariateThreeGroups) {
  Forest<long double, int> forest;

  forest.add_tree(
    std::make_unique<BootstrapTree<long double, int> >(
      std::make_unique<Condition<long double, int> >(
        as_projector({ 0.0, 0.0, 0.0, 0.5982325379690726, -0.8013225508589422 }),
        -0.3483987096124312,
        std::make_unique<Condition<long double, int> >(
          as_projector({ 0.999534397402818, 0.0, -0.030512102657559676, 0.0, 0.0 }),
          5.55339996020167,
          std::make_unique<Response<long double, int> >(1),
          std::make_unique<Response<long double, int> >(2)),
        std::make_unique<Response<long double, int> >(0)
        ))
    );


  forest.add_tree(
    std::make_unique<BootstrapTree<long double, int> >(
      std::make_unique<Condition<long double, int> >(
        as_projector({ 0.9998222455113714, 0.0, -0.018854107791118468, 0.0, 0.0 }),
        5.300417766337716,
        std::make_unique<Condition<long double, int> >(
          as_projector({ 0.9989543519613864, 0.0, 0.0, 0.0457187346435417, 0.0 }),
          1.6094899541803496,
          std::make_unique<Response<long double, int> >(0),
          std::make_unique<Response<long double, int> >(1)),
        std::make_unique<Response<long double, int> >(2)))
    );


  forest.add_tree(
    std::make_unique<BootstrapTree<long double, int> >(
      std::make_unique<Condition<long double, int> >(
        as_projector({ 0.9741975531020036, -0.2256969816591904, 0.0, 0.0, 0.0 }),
        3.9550147456664178,
        std::make_unique<Condition<long double, int> >(
          as_projector({ 0.0, 0.9995561292785718, -0.029791683766431428, 0.0, 0.0 }),
          2.6217629631670403,
          std::make_unique<Response<long double, int> >(0),
          std::make_unique<Response<long double, int> >(1)),
        std::make_unique<Response<long double, int> >(2)))
    );


  forest.add_tree(
    std::make_unique<BootstrapTree<long double, int> >(
      std::make_unique<Condition<long double, int> >(
        as_projector({ 0.9615748657636985, 0.0, 0.0, 0.0, -0.2745428519038971 }),
        4.734758305714628,
        std::make_unique<Condition<long double, int> >(
          as_projector({ 0.0, 0.0, 0.3772334858435029, 0.0, -0.926118187467647 }),
          -0.8315603229605784,
          std::make_unique<Response<long double, int> >(1),
          std::make_unique<Response<long double, int> >(0)),
        std::make_unique<Response<long double, int> >(2)))
    );

  DataColumn<long double> data(5);
  data << 9, 8, 1, 1, 1;

  int result = forest.predict(data);

  ASSERT_EQ(2, result);
}

TEST(Forest, PredictDataSomeVariablesMultivariateThreeGroups) {
  Forest<long double, int> forest;

  forest.add_tree(
    std::make_unique<BootstrapTree<long double, int> >(
      std::make_unique<Condition<long double, int> >(
        as_projector({ 0.0, 0.0, 0.0, 0.5982325379690726, -0.8013225508589422 }),
        -0.3483987096124312,
        std::make_unique<Condition<long double, int> >(
          as_projector({ 0.999534397402818, 0.0, -0.030512102657559676, 0.0, 0.0 }),
          5.55339996020167,
          std::make_unique<Response<long double, int> >(1),
          std::make_unique<Response<long double, int> >(2)),
        std::make_unique<Response<long double, int> >(0))
      )
    );


  forest.add_tree(
    std::make_unique<BootstrapTree<long double, int> >(
      std::make_unique<Condition<long double, int> >(
        as_projector({ 0.9998222455113714, 0.0, -0.018854107791118468, 0.0, 0.0 }),
        5.300417766337716,
        std::make_unique<Condition<long double, int> >(
          as_projector({ 0.9989543519613864, 0.0, 0.0, 0.0457187346435417, 0.0 }),
          1.6094899541803496,
          std::make_unique<Response<long double, int> >(0),
          std::make_unique<Response<long double, int> >(1)),
        std::make_unique<Response<long double, int> >(2))
      )
    );


  forest.add_tree(
    std::make_unique<BootstrapTree<long double, int> >(
      std::make_unique<Condition<long double, int> >(
        as_projector({ 0.9741975531020036, -0.2256969816591904, 0.0, 0.0, 0.0 }),
        3.9550147456664178,
        std::make_unique<Condition<long double, int> >(
          as_projector({ 0.0, 0.9995561292785718, -0.029791683766431428, 0.0, 0.0 }),
          2.6217629631670403,
          std::make_unique<Response<long double, int> >(0),
          std::make_unique<Response<long double, int> >(1)),
        std::make_unique<Response<long double, int> >(2))
      )
    );


  forest.add_tree(
    std::make_unique<BootstrapTree<long double, int> >(
      std::make_unique<Condition<long double, int> >(
        as_projector({ 0.9615748657636985, 0.0, 0.0, 0.0, -0.2745428519038971 }),
        4.734758305714628,
        std::make_unique<Condition<long double, int> >(
          as_projector({ 0.0, 0.0, 0.3772334858435029, 0.0, -0.926118187467647 }),
          -0.8315603229605784,
          std::make_unique<Response<long double, int> >(1),
          std::make_unique<Response<long double, int> >(0)),
        std::make_unique<Response<long double, int> >(2))
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

TEST(Forest, VariableImportanceProjectorLDASomeVariablesMultivariateThreeGroups) {
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
  const int seed = 1;

  Forest<long double, int> forest = Forest<long double, int>::train(
    *TrainingSpec<long double, int>::uniform_glda(n_vars, lambda),
    DataSpec<long double, int>(data, groups),
    4,
    seed);

  DVector<long double> result = forest.variable_importance(VIProjectorStrategy<long double, int>());

  DVector<long double> expected(5);
  expected <<
    0.499640,
    0.249999,
    0.004744,
    0.062054,
    0.064683;

  ASSERT_TRUE(expected.isApprox(result, 0.01)) << std::endl << expected << std::endl << std::endl << result << std::endl;
}

TEST(Forest, VariableImportanceProjectorPDAAllVariablesMultivariateTwoGroups) {
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
  const int seed = 0;

  Forest<long double, int> forest = Forest<long double, int>::train(
    *TrainingSpec<long double, int>::uniform_glda(n_vars, lambda),
    DataSpec<long double, int>(data, groups),
    4,
    seed);

  DVector<long double> result = forest.variable_importance(VIProjectorStrategy<long double, int>());

  Projector<long double> expected = as_projector({
    0.497305,
    0.00889968,
    0.0137289,
    0.0177429,
    0.0126566,
    0.0126566,
    0.0126566,
    0.0126566,
    0.0126566,
    0.0126566,
    0.0126566,
    0.0126566 });

  ASSERT_TRUE(expected.isApprox(result, 0.01));
}

TEST(Forest, VariableImportanceProjectorAdjustedLDASomeVariablesMultivariateThreeGroups) {
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
  const int seed = 1;

  Forest<long double, int> forest = Forest<long double, int>::train(
    *TrainingSpec<long double, int>::uniform_glda(n_vars, lambda),
    DataSpec<long double, int>(data, groups),
    4,
    seed);

  DVector<long double> result = forest.variable_importance(VIProjectorAdjustedStrategy<long double, int>());

  DVector<long double> expected(5);
  expected <<
    0.565137,
    0.247005,
    0.006922,
    0.025883,
    0.035029;


  ASSERT_TRUE(expected.isApprox(result, 0.01)) << std::endl << expected << std::endl << std::endl << result << std::endl;
}

TEST(Forest, VariableImportanceProjectorAdjustedPDAAllVariablesMultivariateTwoGroups) {
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
  const int seed = 0;

  Forest<long double, int> forest = Forest<long double, int>::train(
    *TrainingSpec<long double, int>::uniform_glda(n_vars, lambda),
    DataSpec<long double, int>(data, groups),
    4,
    seed);

  DVector<long double> result = forest.variable_importance(VIProjectorAdjustedStrategy<long double, int>());

  DVector<long double> expected(12);
  expected <<
    0.983637,
    0.018022,
    0.028957,
    0.036786,
    0.026617,
    0.026617,
    0.026617,
    0.026617,
    0.026617,
    0.026617,
    0.026617,
    0.026617;

  ASSERT_TRUE(expected.isApprox(result, 0.01)) << std::endl << expected << std::endl << std::endl << result << std::endl;
}

TEST(Forest, VariableImportancePermutationLDASomeVariablesMultivariateThreeGroups) {
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
  const int seed = 1;

  Forest<long double, int> forest = Forest<long double, int>::train(
    *TrainingSpec<long double, int>::uniform_glda(n_vars, lambda),
    DataSpec<long double, int>(data, groups),
    4,
    seed);

  Random::seed(0);

  DVector<long double> result = forest.variable_importance(VIPermutationStrategy<long double, int>());

  DVector<long double> expected(5);
  expected <<
    0.282954,
    0.125000,
    -0.022727,
    0.093181,
    0.000000;

  ASSERT_TRUE(expected.isApprox(result, 0.01)) << std::endl << expected << std::endl << std::endl << result << std::endl;
}

TEST(Forest, VariableImportancePermutationPDAAllVariablesMultivariateTwoGroups) {
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
  const int seed = 0;

  Forest<long double, int> forest = Forest<long double, int>::train(
    *TrainingSpec<long double, int>::uniform_glda(n_vars, lambda),
    DataSpec<long double, int>(data, groups),
    4,
    seed);

  Random::seed(0);

  DVector<long double> result = forest.variable_importance(VIPermutationStrategy<long double, int>());

  DVector<long double> expected(12);
  expected <<
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0;

  ASSERT_TRUE(expected.isApprox(result, 0.01)) << std::endl << expected << std::endl << std::endl << result << std::endl;
}


TEST(Forest, ErrorRateDataSpecMin) {
  Data<long double> x(30, 5);
  x <<
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

  DataColumn<int> y(30);
  y <<
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

  DataSpec<long double, int> data(x, y);

  const int seed = 0;
  Forest<long double, int> forest = Forest<long double, int>::train(*TrainingSpec<long double, int>::lda(), data, 4, seed);
  DataColumn<int> predictions = forest.predict(data.x);

  long double result = forest.error_rate(DataSpec<long double, int>(x, predictions));

  ASSERT_DOUBLE_EQ(0.0, result);
}

TEST(Forest, ErrorRateDataSpecMax) {
  Data<long double> x(30, 5);
  x <<
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

  DataColumn<int> y(30);
  y <<
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

  DataSpec<long double, int> data(x, y);

  const int seed = 0;
  Forest<long double, int> forest = Forest<long double, int>::train(*TrainingSpec<long double, int>::lda(), data, 4, seed);
  DataColumn<int> predictions = DataColumn<int>::Constant(30, 3);

  long double result = forest.error_rate(DataSpec<long double, int>(x, predictions));

  ASSERT_DOUBLE_EQ(1.0, result);
}

TEST(Forest, ErrorRateDataSpecGeneric) {
  Data<long double> x(30, 5);
  x <<
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

  DataColumn<int> y(30);
  y <<
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

  DataSpec<long double, int> data(x, y);

  const int seed = 0;
  Forest<long double, int> forest = Forest<long double, int>::train(*TrainingSpec<long double, int>::lda(), data, 4, seed);
  DataColumn<int> predictions = DataColumn<int>::Zero(30);

  long double result = forest.error_rate(DataSpec<long double, int>(x, predictions));

  ASSERT_NEAR(0.666, result, 0.1);
}

TEST(Forest, ErrorRateBootstrapDataSpecMin) {
  Data<long double> x(30, 5);
  x <<
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

  DataColumn<int> y(30);
  y <<
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

  DataSpec<long double, int> data(x, y);

  const int seed = 0;
  Forest<long double, int> forest = Forest<long double, int>::train(*TrainingSpec<long double, int>::lda(), data, 4, seed);
  DataColumn<int> predictions = forest.predict(data.x);

  std::vector<int> sample_indices(10);
  std::iota(sample_indices.begin(), sample_indices.end(), 0);

  long double result = forest.error_rate(BootstrapDataSpec<long double, int>(x, predictions, sample_indices));

  ASSERT_DOUBLE_EQ(0.0, result);
}

TEST(Forest, ErrorRateBootstrapDataSpecMax) {
  Data<long double> x(30, 5);
  x <<
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

  DataColumn<int> y(30);
  y <<
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

  DataSpec<long double, int> data(x, y);

  const int seed = 0;
  Forest<long double, int> forest = Forest<long double, int>::train(*TrainingSpec<long double, int>::lda(), data, 4, seed);
  DataColumn<int> predictions = DataColumn<int>::Constant(30, 3);

  std::vector<int> sample_indices(10);
  std::iota(sample_indices.begin(), sample_indices.end(), 0);

  long double result = forest.error_rate(BootstrapDataSpec<long double, int>(x, predictions, sample_indices));

  ASSERT_DOUBLE_EQ(1.0, result);
}

TEST(Forest, ErrorRateBootstrapDataSpecGeneric) {
  Data<long double> x(30, 5);
  x <<
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

  DataColumn<int> y(30);
  y <<
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

  DataSpec<long double, int> data(x, y);

  const int seed = 0;
  Forest<long double, int> forest = Forest<long double, int>::train(*TrainingSpec<long double, int>::lda(), data, 4, seed);
  DataColumn<int> predictions = DataColumn<int>::Zero(30);

  std::vector<int> sample_indices(20);
  std::iota(sample_indices.begin(), sample_indices.end(), 0);

  long double result = forest.error_rate(BootstrapDataSpec<long double, int>(x, predictions, sample_indices));

  ASSERT_NEAR(0.5, result, 0.1);
}

TEST(Forest, ErrorRate) {
  Data<long double> x(30, 5);
  x <<
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

  DataColumn<int> y(30);
  y <<
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

  DataSpec<long double, int> data(x, y);

  const int seed = 0;
  Forest<long double, int> forest = Forest<long double, int>::train(*TrainingSpec<long double, int>::lda(), data, 4, seed);

  long double result = forest.error_rate();

  ASSERT_NEAR(0.0, result, 0.1);
}

TEST(Forest, ConfusionMatrixDataSpecDiagonal) {
  Data<long double> x(30, 5);
  x <<
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

  DataColumn<int> y(30);
  y <<
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

  DataSpec<long double, int> data(x, y);

  const int seed = 0;
  Forest<long double, int> forest = Forest<long double, int>::train(*TrainingSpec<long double, int>::lda(), data, 4, seed);
  DataColumn<int> predictions = forest.predict(data.x);

  ConfusionMatrix result = forest.confusion_matrix(DataSpec<long double, int>(x, predictions));

  Data<int> expected = Data<int>::Zero(3, 3);
  expected.diagonal() << 10, 12, 8;

  ASSERT_EQ(expected.size(), result.values.size());
  ASSERT_EQ(expected.rows(), result.values.rows());
  ASSERT_EQ(expected.cols(), result.values.cols());
  ASSERT_EQ(expected, result.values);

  ASSERT_EQ((std::map<int, int>({ { 0, 0 }, { 1, 1 }, { 2, 2 } })), result.label_index);
}

TEST(Forest, ConfusionMatrixDataSpecZeroDiagonal) {
  Data<long double> x(30, 5);
  x <<
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

  DataColumn<int> y(30);
  y <<
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

  DataSpec<long double, int> data(x, y);

  const int seed = 0;
  Forest<long double, int> forest = Forest<long double, int>::train(*TrainingSpec<long double, int>::lda(), data, 4, seed);

  DataColumn<int> predictions(30);
  predictions <<
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
    2,
    2,
    2,
    2,
    2,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0;

  ConfusionMatrix result = forest.confusion_matrix(DataSpec<long double, int>(x, predictions));

  Data<int> expected(3, 3);
  expected <<
    0, 0, 8,
    10, 0, 0,
    0, 12, 0;

  ASSERT_EQ(expected.size(), result.values.size());
  ASSERT_EQ(expected.rows(), result.values.rows());
  ASSERT_EQ(expected.cols(), result.values.cols());
  ASSERT_EQ(expected, result.values);

  ASSERT_EQ((std::map<int, int>({ { 0, 0 }, { 1, 1 }, { 2, 2 } })), result.label_index);
}

TEST(Forest, ConfusionMatrixBootstrapDataSpecDiagonal) {
  Data<long double> x(30, 5);
  x <<
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

  DataColumn<int> y(30);
  y <<
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

  DataSpec<long double, int> data(x, y);

  const int seed = 0;
  Forest<long double, int> forest = Forest<long double, int>::train(*TrainingSpec<long double, int>::lda(), data, 4, seed);
  DataColumn<int> predictions = forest.predict(data.x);

  std::vector<int> sample_indices = { 0, 1, 2, 3, 13, 14, 15, 16, 26, 27, 28, 29 };

  ConfusionMatrix result = forest.confusion_matrix(BootstrapDataSpec<long double, int>(x, predictions, sample_indices));

  Data<int> expected = Data<int>::Zero(3, 3);
  expected.diagonal() << 4, 4, 4;

  ASSERT_EQ(expected.size(), result.values.size());
  ASSERT_EQ(expected.rows(), result.values.rows());
  ASSERT_EQ(expected.cols(), result.values.cols());
  ASSERT_EQ(expected, result.values);

  ASSERT_EQ((std::map<int, int>({ { 0, 0 }, { 1, 1 }, { 2, 2 } })), result.label_index);
}

TEST(Forest, ConfusionMatrixBootstrapDataSpecZeroDiagonal) {
  Data<long double> x(30, 5);
  x <<
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

  DataColumn<int> y(30);
  y <<
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

  DataSpec<long double, int> data(x, y);

  const int seed = 0;
  Forest<long double, int> forest = Forest<long double, int>::train(*TrainingSpec<long double, int>::lda(), data, 4, seed);

  DataColumn<int> predictions(30);
  predictions <<
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
    2,
    2,
    2,
    2,
    2,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0;

  ConfusionMatrix result = forest.confusion_matrix(DataSpec<long double, int>(x, predictions));

  Data<int> expected(3, 3);
  expected <<
    0, 0, 8,
    10, 0, 0,
    0, 12, 0;

  ASSERT_EQ(expected.size(), result.values.size());
  ASSERT_EQ(expected.rows(), result.values.rows());
  ASSERT_EQ(expected.cols(), result.values.cols());
  ASSERT_EQ(expected, result.values);

  ASSERT_EQ((std::map<int, int>({ { 0, 0 }, { 1, 1 }, { 2, 2 } })), result.label_index);
}

TEST(Forest, ConfusionMatrix) {
  Data<long double> x(30, 5);
  x <<
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

  DataColumn<int> y(30);
  y <<
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

  DataSpec<long double, int> data(x, y);

  const int seed = 0;
  Forest<long double, int> forest = Forest<long double, int>::train(*TrainingSpec<long double, int>::lda(), data, 4, seed);

  ConfusionMatrix result = forest.confusion_matrix();

  Data<int> expected(3, 3);
  expected <<
    9, 0, 0,
    0, 10, 0,
    0, 0, 6;

  ASSERT_EQ(expected.size(), result.values.size());
  ASSERT_EQ(expected.rows(), result.values.rows());
  ASSERT_EQ(expected.cols(), result.values.cols());
  ASSERT_EQ(expected, result.values);

  ASSERT_EQ((std::map<int, int>({ { 0, 0 }, { 1, 1 }, { 2, 2 } })), result.label_index);
}
