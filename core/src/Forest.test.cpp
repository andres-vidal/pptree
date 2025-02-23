#include <gtest/gtest.h>

#include "Forest.hpp"
#include "VIStrategy.hpp"

#include "Macros.hpp"

using namespace models;
using namespace models::stats;
using namespace models::pp;
using namespace models::math;

static Projector<float> as_projector(std::vector<float> vector) {
  Eigen::Map<Projector<float> > projector(vector.data(), vector.size());
  return projector;
}

TEST(Forest, TrainLDAAllVariablesMultivariateThreeGroups) {
  Data<float> data(30, 5);
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
  const float lambda = 0;
  const int seed = 0;


  Forest<float, int> result = Forest<float, int>::train(
    *TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
    SortedDataSpec<float, int>(data, groups),
    4,
    seed);

  Forest<float, int> expect(
    TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
    std::make_shared<SortedDataSpec<float, int> >(data, groups, std::set<int>({ 0, 1, 2 })),
    seed);

  expect.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      std::make_unique<Condition<float, int> >(
        as_projector({ 0.3959386339593606, -0.908269881092349, -0.05734470673819268, 0.08786184419030313, -0.0852660670107132 }),
        -1.7110096455357062,
        std::make_unique<Condition<float, int> >(
          as_projector({ -1.803617104793913e-15, 1.0, -0.0, 0.0, 0.0 }),
          6.49999999999999,
          std::make_unique<Response<float, int> >(1),
          std::make_unique<Response<float, int> >(2)
          ),
        std::make_unique<Response<float, int> >(0)
        ),
      TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
      std::make_shared<BootstrapDataSpec<float, int> >(data, groups, std::set<int>({ 0, 1, 2 }), std::vector<int>({ 0, 1 })))
    );

  expect.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      std::make_unique<Condition<float, int> >(
        as_projector({ 0.9429063113601566, -0.1791114759384774, -0.1577467223900484, 0.07803343725612998, -0.21880018608199273 }),
        3.745534637999798,
        std::make_unique<Condition<float, int> >(
          as_projector({ 0.09705833399783349, 0.9801862846848319, -0.027094022564766305, -0.005144434664196154, 0.17045226854036435 }),
          2.8344748952402568,
          std::make_unique<Response<float, int> >(0),
          std::make_unique<Response<float, int> >(1)
          ),
        std::make_unique<Response<float, int> >(2)
        ),
      TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
      std::make_shared<BootstrapDataSpec<float, int> >(data, groups, std::set<int>({ 0, 1, 2 }), std::vector<int>({ 0, 1 })))
    );


  expect.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      std::make_unique<Condition<float, int> >(
        as_projector({ 0.9541878806121409, -0.1693811442451475, -0.11681578122561698, 0.07761527693154766, -0.20289272660843075 }),
        3.8815126994794134,
        std::make_unique<Condition<float, int> >(
          as_projector({ 0.07964113289220642, 0.9872406081485949, -0.019516630829984314, 0.04771950247256534, 0.1278875356665199 }),
          2.8098223049798854,
          std::make_unique<Response<float, int> >(0),
          std::make_unique<Response<float, int> >(1)
          ),
        std::make_unique<Response<float, int> >(2)
        ),
      TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
      std::make_shared<BootstrapDataSpec<float, int> >(data, groups, std::set<int>({ 0, 1, 2 }), std::vector<int>({ 0, 1 })))

    );


  expect.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      std::make_unique<Condition<float, int> >(
        as_projector({ 0.9774257025571, -0.20095147055468665, 0.012604451134322587, -0.015914730117438724, -0.06201089936099892 }),
        3.9924452660765164,
        std::make_unique<Condition<float, int> >(
          as_projector({ 1.0, 2.0286166352810033e-15, -0.0, -0.0, -0.0 }),
          1.5000000000000053,
          std::make_unique<Response<float, int> >(0),
          std::make_unique<Response<float, int> >(1)
          ),
        std::make_unique<Response<float, int> >(2)
        ),
      TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
      std::make_shared<BootstrapDataSpec<float, int> >(data, groups, std::set<int>({ 0, 1, 2 }), std::vector<int>({ 0, 1 })))

    );



  ASSERT_EQ(expect, result);
  ASSERT_EQ(expect.seed, result.seed);
}

TEST(Forest, TrainLDASomeVariablesMultivariateThreeGroups) {
  Data<float> data(30, 5);
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
  const float lambda = 0;
  const int seed = 1;

  Forest<float, int> result = Forest<float, int>::train(
    *TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
    SortedDataSpec<float, int>(data, groups),
    4,
    seed);

  Forest<float, int> expect(
    TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
    std::make_shared<SortedDataSpec<float, int> >(data, groups, std::set<int>({ 0, 1, 2 })),
    seed);

  expect.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      std::make_unique<Condition<float, int> >(
        as_projector({ 0.9758559465408325, -0.09231313318014145, -0.0032241533044725657, -0.18500669300556183, 0.0703246220946312 }),
        4.111116409301758,
        std::make_unique<Condition<float, int> >(
          as_projector({ 0.15599919855594635, 0.9833118319511414, 0.0935513898730278, 0.002791941398754716, 0.001575765199959278 }),
          2.8408889770507813,
          std::make_unique<Response<float, int> >(0),
          std::make_unique<Response<float, int> >(1)
          ),
        std::make_unique<Response<float, int> >(2)
        ),
      TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
      std::make_shared<BootstrapDataSpec<float, int> >(
        data, groups, std::set<int>({ 0, 1, 2 }), std::vector<int>({ 0, 1 }))
      )
    );

  expect.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      std::make_unique<Condition<float, int> >(
        as_projector({ 0.9792769551277161, -0.19746960699558258, 0.006554579362273216, -0.04340337961912155, -0.00977456383407116 }),
        4.042616367340088,
        std::make_unique<Condition<float, int> >(
          as_projector({ 1.0, 6.720548384464564e-08, 7.547996005996538e-09, -1.461530541035927e-08, 2.5287971539000864e-07 }),
          1.5000007152557373,
          std::make_unique<Response<float, int> >(0),
          std::make_unique<Response<float, int> >(1)
          ),
        std::make_unique<Response<float, int> >(2)
        ),
      TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
      std::make_shared<BootstrapDataSpec<float, int> >(
        data, groups, std::set<int>({ 0, 1, 2 }), std::vector<int>({ 0, 1 }))
      )
    );

  expect.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      std::make_unique<Condition<float, int> >(
        as_projector({ 0.9805459976196289, -0.19516022503376007, -0.004170051775872707, -0.02060389146208763, 0.0003253696486353874 }),
        4.093138694763184,
        std::make_unique<Condition<float, int> >(
          as_projector({ 1.0, 1.299416112487961e-06, -2.8670829266275177e-08, 1.2065898147284315e-07, 2.986283575978632e-08 }),
          1.500003457069397,
          std::make_unique<Response<float, int> >(0),
          std::make_unique<Response<float, int> >(1)
          ),
        std::make_unique<Response<float, int> >(2)
        ),
      TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
      std::make_shared<BootstrapDataSpec<float, int> >(
        data, groups, std::set<int>({ 0, 1, 2 }), std::vector<int>({ 0, 1 }))
      )
    );

  expect.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      std::make_unique<Condition<float, int> >(
        as_projector({ 0.16805875301361084, 0.18092086911201477, 0.05232581868767738, -0.9623227715492249, -0.10109801590442657 }),
        -3.9484755992889404,
        std::make_unique<Response<float, int> >(2),
        std::make_unique<Condition<float, int> >(
          as_projector({ 0.11039015650749207, 0.9367238283157349, 0.006573501508682966, 0.30068090558052063, 0.1411038339138031 }),
          3.1110880374908447,
          std::make_unique<Response<float, int> >(0),
          std::make_unique<Response<float, int> >(1)
          )
        ),
      TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
      std::make_shared<BootstrapDataSpec<float, int> >(
        data, groups, std::set<int>({ 0, 1, 2 }), std::vector<int>({ 0, 1 }))
      )
    );

  ASSERT_EQ(expect, result);
  ASSERT_EQ(expect.seed, result.seed);
}

TEST(Forest, TrainPDAAllVariablesMultivariateTwoGroups) {
  Data<float> data(10, 12);
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
  const float lambda = 0.1;
  const int seed = 0;

  Forest<float, int> result = Forest<float, int>::train(
    *TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
    SortedDataSpec<float, int>(data, groups),
    4,
    seed);


  Forest<float, int> expect(
    TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
    std::make_shared<SortedDataSpec<float, int> >(data, groups, std::set<int>({ 0, 1, 2 })),
    seed);

  expect.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      std::make_unique<Condition<float, int> >(
        as_projector({ 0.9655843123155974, 0.08681796476306242, 0.09953066811975722, -0.1290329926431043, -0.06476274943747946, -0.0647627494374795, -0.06476274943747948, -0.06476274943747948, -0.06476274943747941, -0.06476274943747942, -0.06476274943747944, -0.06476274943747942 }),
        1.66054084868256,
        std::make_unique<Response<float, int> >(0),
        std::make_unique<Response<float, int> >(1)
        ),
      TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
      std::make_shared<BootstrapDataSpec<float, int> >(data, groups, std::set<int>({ 0, 1, 2 }), std::vector<int>({ 0, 1 })))
    );

  expect.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      std::make_unique<Condition<float, int> >(
        as_projector({ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }),
        2.5,
        std::make_unique<Response<float, int> >(0),
        std::make_unique<Response<float, int> >(1)),
      TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
      std::make_shared<BootstrapDataSpec<float, int> >(data, groups, std::set<int>({ 0, 1, 2 }), std::vector<int>({ 0, 1 })))
    );

  expect.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      std::make_unique<Condition<float, int> >(
        as_projector({ 0.9607071714826417, 0.13071424952758612, 0.18649074813826336, -0.06675108245672948, -0.05089328231941315, -0.05089328231941322, -0.05089328231941318, -0.05089328231941322, -0.05089328231941308, -0.050893282319413084, -0.05089328231941308, -0.0508932823194131 }),
        1.9601144528047953,
        std::make_unique<Response<float, int> >(0),
        std::make_unique<Response<float, int> >(1)),
      TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
      std::make_shared<BootstrapDataSpec<float, int> >(data, groups, std::set<int>({ 0, 1, 2 }), std::vector<int>({ 0, 1 })))
    );

  expect.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      std::make_unique<Condition<float, int> >(
        as_projector({ 0.9611377266908191, 0.027169137836983416, 0.08147067425776834, -0.05364933895611067, -0.09080224800793527, -0.0908022480079353, -0.0908022480079353, -0.09080224800793528, -0.09080224800793517, -0.09080224800793522, -0.09080224800793522, -0.09080224800793524 }),
        1.3347158081570496,
        std::make_unique<Response<float, int> >(0),
        std::make_unique<Response<float, int> >(1)),
      TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
      std::make_shared<BootstrapDataSpec<float, int> >(data, groups, std::set<int>({ 0, 1, 2 }), std::vector<int>({ 0, 1 })))
    );

  ASSERT_EQ(expect, result);
  ASSERT_EQ(expect.seed, result.seed);
}

TEST(Forest, PredictDataColumnSomeVariablesMultivariateThreeGroups) {
  Forest<float, int> forest;

  forest.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      std::make_unique<Condition<float, int> >(
        as_projector({ 0.0, 0.0, 0.0, 0.5982325379690726, -0.8013225508589422 }),
        -0.3483987096124312,
        std::make_unique<Condition<float, int> >(
          as_projector({ 0.999534397402818, 0.0, -0.030512102657559676, 0.0, 0.0 }),
          5.55339996020167,
          std::make_unique<Response<float, int> >(1),
          std::make_unique<Response<float, int> >(2)),
        std::make_unique<Response<float, int> >(0)
        ))
    );


  forest.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      std::make_unique<Condition<float, int> >(
        as_projector({ 0.9998222455113714, 0.0, -0.018854107791118468, 0.0, 0.0 }),
        5.300417766337716,
        std::make_unique<Condition<float, int> >(
          as_projector({ 0.9989543519613864, 0.0, 0.0, 0.0457187346435417, 0.0 }),
          1.6094899541803496,
          std::make_unique<Response<float, int> >(0),
          std::make_unique<Response<float, int> >(1)),
        std::make_unique<Response<float, int> >(2)))
    );


  forest.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      std::make_unique<Condition<float, int> >(
        as_projector({ 0.9741975531020036, -0.2256969816591904, 0.0, 0.0, 0.0 }),
        3.9550147456664178,
        std::make_unique<Condition<float, int> >(
          as_projector({ 0.0, 0.9995561292785718, -0.029791683766431428, 0.0, 0.0 }),
          2.6217629631670403,
          std::make_unique<Response<float, int> >(0),
          std::make_unique<Response<float, int> >(1)),
        std::make_unique<Response<float, int> >(2)))
    );


  forest.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      std::make_unique<Condition<float, int> >(
        as_projector({ 0.9615748657636985, 0.0, 0.0, 0.0, -0.2745428519038971 }),
        4.734758305714628,
        std::make_unique<Condition<float, int> >(
          as_projector({ 0.0, 0.0, 0.3772334858435029, 0.0, -0.926118187467647 }),
          -0.8315603229605784,
          std::make_unique<Response<float, int> >(1),
          std::make_unique<Response<float, int> >(0)),
        std::make_unique<Response<float, int> >(2)))
    );

  DataColumn<float> data(5);
  data << 9, 8, 1, 1, 1;

  int result = forest.predict(data);

  ASSERT_EQ(2, result);
}

TEST(Forest, PredictDataSomeVariablesMultivariateThreeGroups) {
  Forest<float, int> forest;

  forest.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      std::make_unique<Condition<float, int> >(
        as_projector({ 0.0, 0.0, 0.0, 0.5982325379690726, -0.8013225508589422 }),
        -0.3483987096124312,
        std::make_unique<Condition<float, int> >(
          as_projector({ 0.999534397402818, 0.0, -0.030512102657559676, 0.0, 0.0 }),
          5.55339996020167,
          std::make_unique<Response<float, int> >(1),
          std::make_unique<Response<float, int> >(2)),
        std::make_unique<Response<float, int> >(0))
      )
    );


  forest.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      std::make_unique<Condition<float, int> >(
        as_projector({ 0.9998222455113714, 0.0, -0.018854107791118468, 0.0, 0.0 }),
        5.300417766337716,
        std::make_unique<Condition<float, int> >(
          as_projector({ 0.9989543519613864, 0.0, 0.0, 0.0457187346435417, 0.0 }),
          1.6094899541803496,
          std::make_unique<Response<float, int> >(0),
          std::make_unique<Response<float, int> >(1)),
        std::make_unique<Response<float, int> >(2))
      )
    );


  forest.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      std::make_unique<Condition<float, int> >(
        as_projector({ 0.9741975531020036, -0.2256969816591904, 0.0, 0.0, 0.0 }),
        3.9550147456664178,
        std::make_unique<Condition<float, int> >(
          as_projector({ 0.0, 0.9995561292785718, -0.029791683766431428, 0.0, 0.0 }),
          2.6217629631670403,
          std::make_unique<Response<float, int> >(0),
          std::make_unique<Response<float, int> >(1)),
        std::make_unique<Response<float, int> >(2))
      )
    );


  forest.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      std::make_unique<Condition<float, int> >(
        as_projector({ 0.9615748657636985, 0.0, 0.0, 0.0, -0.2745428519038971 }),
        4.734758305714628,
        std::make_unique<Condition<float, int> >(
          as_projector({ 0.0, 0.0, 0.3772334858435029, 0.0, -0.926118187467647 }),
          -0.8315603229605784,
          std::make_unique<Response<float, int> >(1),
          std::make_unique<Response<float, int> >(0)),
        std::make_unique<Response<float, int> >(2))
      )
    );

  Data<float> data(30, 5);
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
  Data<float> data(30, 5);
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
  const float lambda = 0;
  const int seed = 1;

  Forest<float, int> forest = Forest<float, int>::train(
    *TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
    SortedDataSpec<float, int>(data, groups),
    4,
    seed);

  DVector<float> result = forest.variable_importance(VIProjectorStrategy<float, int>());

  DVector<float> expected(5);
  expected <<
    0.16962195932865143,
    0.26724648475646973,
    0.00253944098949432,
    0.36136019229888916,
    0.14448873698711395;

  ASSERT_APPROX(expected, result);
}

TEST(Forest, VariableImportanceProjectorPDAAllVariablesMultivariateTwoGroups) {
  Data<float> data(10, 12);
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
  const float lambda = 0.1;
  const int seed = 0;

  Forest<float, int> forest = Forest<float, int>::train(
    *TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
    SortedDataSpec<float, int>(data, groups),
    4,
    seed);

  DVector<float> result = forest.variable_importance(VIProjectorStrategy<float, int>());

  Projector<float> expected = as_projector({
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

  ASSERT_APPROX(expected, result);
}

TEST(Forest, VariableImportanceProjectorAdjustedLDASomeVariablesMultivariateThreeGroups) {
  Data<float> data(30, 5);
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
  const float lambda = 0;
  const int seed = 1;

  Forest<float, int> forest = Forest<float, int>::train(
    *TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
    SortedDataSpec<float, int>(data, groups),
    4,
    seed);

  DVector<float> result = forest.variable_importance(VIProjectorAdjustedStrategy<float, int>());

  DVector<float> expected(5);
  expected <<
    0.19476625323295593,
    0.18999017775058746,
    0.00231075938791036,
    0.17675185203552246,
    0.06259537488222122;


  ASSERT_APPROX(expected, result);
}

TEST(Forest, VariableImportanceProjectorAdjustedPDAAllVariablesMultivariateTwoGroups) {
  Data<float> data(10, 12);
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
  const float lambda = 0.1;
  const int seed = 0;

  Forest<float, int> forest = Forest<float, int>::train(
    *TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
    SortedDataSpec<float, int>(data, groups),
    4,
    seed);

  DVector<float> result = forest.variable_importance(VIProjectorAdjustedStrategy<float, int>());

  DVector<float> expected(12);
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

  ASSERT_APPROX(expected, result);
}

TEST(Forest, VariableImportancePermutationLDASomeVariablesMultivariateThreeGroups) {
  Data<float> data(30, 5);
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
  const float lambda = 0;
  const int seed = 1;

  Forest<float, int> forest = Forest<float, int>::train(
    *TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
    SortedDataSpec<float, int>(data, groups),
    4,
    seed);

  Random::seed(0);

  DVector<float> result = forest.variable_importance(VIPermutationStrategy<float, int>());

  DVector<float> expected(5);
  expected <<
    0.34999999403953552,
    0.20000000298023224,
    0,
    0,
    0;


  ASSERT_APPROX(expected, result);
}

TEST(Forest, VariableImportancePermutationPDAAllVariablesMultivariateTwoGroups) {
  Data<float> data(10, 12);
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
  const float lambda = 0.1;
  const int seed = 0;

  Forest<float, int> forest = Forest<float, int>::train(
    *TrainingSpec<float, int>::uniform_glda(n_vars, lambda),
    SortedDataSpec<float, int>(data, groups),
    4,
    seed);

  Random::seed(0);

  DVector<float> result = forest.variable_importance(VIPermutationStrategy<float, int>());

  DVector<float> expected(12);
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

  ASSERT_APPROX(expected, result);
}


TEST(Forest, ErrorRateDataSpecMin) {
  Data<float> x(30, 5);
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

  SortedDataSpec<float, int> data(x, y);

  const int seed = 0;
  Forest<float, int> forest = Forest<float, int>::train(*TrainingSpec<float, int>::lda(), data, 4, seed);
  DataColumn<int> predictions = forest.predict(data.x);

  float result = forest.error_rate(SortedDataSpec<float, int>(x, predictions));

  ASSERT_FLOAT_EQ(0.0, result);
}

TEST(Forest, ErrorRateDataSpecMax) {
  Data<float> x(30, 5);
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

  SortedDataSpec<float, int> data(x, y);

  const int seed = 0;
  Forest<float, int> forest = Forest<float, int>::train(*TrainingSpec<float, int>::lda(), data, 4, seed);
  DataColumn<int> predictions = DataColumn<int>::Constant(30, 3);

  float result = forest.error_rate(SortedDataSpec<float, int>(x, predictions));

  ASSERT_FLOAT_EQ(1.0, result);
}

TEST(Forest, ErrorRateDataSpecGeneric) {
  Data<float> x(30, 5);
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

  SortedDataSpec<float, int> data(x, y);

  const int seed = 0;
  Forest<float, int> forest = Forest<float, int>::train(*TrainingSpec<float, int>::lda(), data, 4, seed);
  DataColumn<int> predictions = DataColumn<int>::Zero(30);

  float result = forest.error_rate(SortedDataSpec<float, int>(x, predictions));

  ASSERT_NEAR(0.666, result, 0.1);
}

TEST(Forest, ErrorRateBootstrapDataSpecMin) {
  Data<float> x(30, 5);
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

  SortedDataSpec<float, int> data(x, y);

  const int seed = 0;
  Forest<float, int> forest = Forest<float, int>::train(*TrainingSpec<float, int>::lda(), data, 4, seed);
  DataColumn<int> predictions = forest.predict(data.x);

  std::vector<int> sample_indices(10);
  std::iota(sample_indices.begin(), sample_indices.end(), 0);

  float result = forest.error_rate(BootstrapDataSpec<float, int>(x, predictions, sample_indices));

  ASSERT_FLOAT_EQ(0.0, result);
}

TEST(Forest, ErrorRateBootstrapDataSpecMax) {
  Data<float> x(30, 5);
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

  SortedDataSpec<float, int> data(x, y);

  const int seed = 0;
  Forest<float, int> forest = Forest<float, int>::train(*TrainingSpec<float, int>::lda(), data, 4, seed);
  DataColumn<int> predictions = DataColumn<int>::Constant(30, 3);

  std::vector<int> sample_indices(10);
  std::iota(sample_indices.begin(), sample_indices.end(), 0);

  float result = forest.error_rate(BootstrapDataSpec<float, int>(x, predictions, sample_indices));

  ASSERT_FLOAT_EQ(1.0, result);
}

TEST(Forest, ErrorRateBootstrapDataSpecGeneric) {
  Data<float> x(30, 5);
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

  SortedDataSpec<float, int> data(x, y);

  const int seed = 0;
  Forest<float, int> forest = Forest<float, int>::train(*TrainingSpec<float, int>::lda(), data, 4, seed);
  DataColumn<int> predictions = DataColumn<int>::Zero(30);

  std::vector<int> sample_indices(20);
  std::iota(sample_indices.begin(), sample_indices.end(), 0);

  float result = forest.error_rate(BootstrapDataSpec<float, int>(x, predictions, sample_indices));

  ASSERT_NEAR(0.5, result, 0.1);
}

TEST(Forest, ErrorRate) {
  Data<float> x(30, 5);
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

  SortedDataSpec<float, int> data(x, y);

  const int seed = 0;
  Forest<float, int> forest = Forest<float, int>::train(*TrainingSpec<float, int>::lda(), data, 4, seed);

  float result = forest.error_rate();

  ASSERT_NEAR(0.0, result, 0.1);
}

TEST(Forest, ConfusionMatrixDataSpecDiagonal) {
  Data<float> x(30, 5);
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

  SortedDataSpec<float, int> data(x, y);

  const int seed = 0;
  Forest<float, int> forest = Forest<float, int>::train(*TrainingSpec<float, int>::lda(), data, 4, seed);
  DataColumn<int> predictions = forest.predict(data.x);

  ConfusionMatrix result = forest.confusion_matrix(SortedDataSpec<float, int>(x, predictions));

  Data<int> expected = Data<int>::Zero(3, 3);
  expected.diagonal() << 10, 12, 8;

  ASSERT_EQ(expected.size(), result.values.size());
  ASSERT_EQ(expected.rows(), result.values.rows());
  ASSERT_EQ(expected.cols(), result.values.cols());
  ASSERT_EQ(expected, result.values);

  ASSERT_EQ((std::map<int, int>({ { 0, 0 }, { 1, 1 }, { 2, 2 } })), result.label_index);
}

TEST(Forest, ConfusionMatrixDataSpecZeroDiagonal) {
  Data<float> x(30, 5);
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

  SortedDataSpec<float, int> data(x, y);

  const int seed = 0;
  Forest<float, int> forest = Forest<float, int>::train(*TrainingSpec<float, int>::lda(), data, 4, seed);

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

  ConfusionMatrix result = forest.confusion_matrix(SortedDataSpec<float, int>(x, predictions));

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
  Data<float> x(30, 5);
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

  SortedDataSpec<float, int> data(x, y);

  const int seed = 0;
  Forest<float, int> forest = Forest<float, int>::train(*TrainingSpec<float, int>::lda(), data, 4, seed);
  DataColumn<int> predictions = forest.predict(data.x);

  std::vector<int> sample_indices = { 0, 1, 2, 3, 13, 14, 15, 16, 26, 27, 28, 29 };

  ConfusionMatrix result = forest.confusion_matrix(BootstrapDataSpec<float, int>(x, predictions, sample_indices));

  Data<int> expected = Data<int>::Zero(3, 3);
  expected.diagonal() << 4, 4, 4;

  ASSERT_EQ(expected.size(), result.values.size());
  ASSERT_EQ(expected.rows(), result.values.rows());
  ASSERT_EQ(expected.cols(), result.values.cols());
  ASSERT_EQ(expected, result.values);

  ASSERT_EQ((std::map<int, int>({ { 0, 0 }, { 1, 1 }, { 2, 2 } })), result.label_index);
}

TEST(Forest, ConfusionMatrixBootstrapDataSpecZeroDiagonal) {
  Data<float> x(30, 5);
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

  SortedDataSpec<float, int> data(x, y);

  const int seed = 0;
  Forest<float, int> forest = Forest<float, int>::train(*TrainingSpec<float, int>::lda(), data, 4, seed);

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

  ConfusionMatrix result = forest.confusion_matrix(SortedDataSpec<float, int>(x, predictions));

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
  Data<float> x(30, 5);
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

  SortedDataSpec<float, int> data(x, y);

  const int seed = 0;
  Forest<float, int> forest = Forest<float, int>::train(*TrainingSpec<float, int>::lda(), data, 4, seed);

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
