#include <gtest/gtest.h>

#include "Forest.hpp"

#include "TrainingSpec.hpp"
#include "TrainingSpecGLDA.hpp"
#include "TrainingSpecUGLDA.hpp"

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

  const int n_vars   = x.cols();
  const float lambda = 0;
  const int seed     = 0;

  Forest<float, int> result = Forest<float, int>::train(
    TrainingSpecUGLDA<float, int>(n_vars, lambda),
    x,
    y,
    4,
    seed);

  Forest<float, int> expect;

  expect.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      TreeCondition<float, int>::make(
        as_projector({ 0.9580563306808472, -0.1769358515739441, 0.006788954604417086, 0.10231059044599533, -0.2007693201303482 }),
        3.9684371948242188,
        TreeCondition<float, int>::make(
          as_projector({ 0.03719514608383179, 0.9892118573188782, -0.03843306005001068, 0.03852792829275131, 0.13082440197467804 }),
          2.7188374996185303,
          TreeResponse<float, int>::make(0),
          TreeResponse<float, int>::make(1)
          ),
        TreeResponse<float, int>::make(2)
        ))
    );

  expect.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      TreeCondition<float, int>::make(
        as_projector({ 0.39588773250579834, -0.912018895149231, -0.02612934447824955, 0.07246564328670502, -0.07456832379102707 }),
        -1.714177131652832,
        TreeCondition<float, int>::make(
          as_projector({ -0.0, 1.0, -1.2741912559788293e-11, 5.284577636599508e-11, -2.1893145282780857e-11 }),
          6.5,
          TreeResponse<float, int>::make(1),
          TreeResponse<float, int>::make(2)
          ),
        TreeResponse<float, int>::make(0)
        ))
    );

  expect.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      TreeCondition<float, int>::make(
        as_projector({ 0.9647202491760254, -0.17297711968421936, -0.01983051560819149, 0.12365056574344635, -0.15398375689983368 }),
        4.064751625061035,
        TreeCondition<float, int>::make(
          as_projector({ 0.05080557242035866, 0.9409289360046387, 0.11455754935741425, -0.1627458930015564, 0.2691875696182251 }),
          2.7592363357543945,
          TreeResponse<float, int>::make(0),
          TreeResponse<float, int>::make(1)
          ),
        TreeResponse<float, int>::make(2)
        ))
    );

  expect.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      TreeCondition<float, int>::make(
        as_projector({ 0.39640840888023376, -0.9168400168418884, -0.015512045472860336, 0.027026182040572166, -0.0359681062400341 }),
        -1.7061063051223755,
        TreeCondition<float, int>::make(
          as_projector({ -0.0, 1.0, -2.361130313488502e-08, 6.467465230031166e-09, 1.661359050331157e-08 }),
          6.5,
          TreeResponse<float, int>::make(1),
          TreeResponse<float, int>::make(2)
          ),
        TreeResponse<float, int>::make(0)
        ))
    );

  ASSERT_EQ(expect, result);
  ASSERT_EQ(expect.seed, result.seed);
}

TEST(Forest, TrainLDASomeVariablesMultivariateThreeGroups) {
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


  const int n_vars   = 2;
  const float lambda = 0;
  const int seed     = 1;

  Forest<float, int> result = Forest<float, int>::train(
    TrainingSpecUGLDA<float, int>(n_vars, lambda),
    x,
    y,
    4,
    seed);

  Forest<float, int> expect;

  // First tree
  expect.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      TreeCondition<float, int>::make(
        as_projector({ 0.9023476243019104, 0.0, 0.0, 0.4310089647769928, 0.0 }),
        5.072210788726807,
        TreeCondition<float, int>::make(
          as_projector({ 0.0, 0.9889134764671326, 0.0, 0.14849308133125305, 0.0 }),
          2.7555017471313477,
          TreeResponse<float, int>::make(0),
          TreeResponse<float, int>::make(1)
          ),
        TreeResponse<float, int>::make(2)
        ))
    );

  // Second tree
  expect.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      TreeCondition<float, int>::make(
        as_projector({ -0.9785193204879761, 0.0, 0.0, 0.0, 0.20615507662296295 }),
        -4.900600433349609,
        TreeResponse<float, int>::make(2),
        TreeCondition<float, int>::make(
          as_projector({ 0.0, 0.0, 0.0, 0.477281779050827, -0.8787502646446228 }),
          -0.4522837996482849,
          TreeResponse<float, int>::make(1),
          TreeResponse<float, int>::make(0)
          ))
      )
    );

  // Third tree
  expect.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      TreeCondition<float, int>::make(
        as_projector({ 0.9749178290367126, -0.22256524860858917, 0.0, 0.0, 0.0 }),
        3.98091459274292,
        TreeCondition<float, int>::make(
          as_projector({ 0.0, 0.0, 0.0, 0.16338759660720825, -0.9865618944168091 }),
          -0.7777230739593506,
          TreeResponse<float, int>::make(1),
          TreeResponse<float, int>::make(0)
          ),
        TreeResponse<float, int>::make(2)
        ))
    );

  // Fourth tree
  expect.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      TreeCondition<float, int>::make(
        as_projector({ 0.0, 0.0, 0.0, -0.6386693716049194, 0.7694813013076782 }),
        0.23969349265098572,
        TreeResponse<float, int>::make(0),
        TreeCondition<float, int>::make(
          as_projector({ 0.0, 1.0, 0.0, 0.0, 0.0 }),
          6.5,
          TreeResponse<float, int>::make(1),
          TreeResponse<float, int>::make(2)
          ))
      )
    );

  ASSERT_EQ(expect, result);
  ASSERT_EQ(seed, result.seed);
}

TEST(Forest, TrainPDAAllVariablesMultivariateTwoGroups) {
  Data<float> x(10, 12);
  x <<
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

  DataColumn<int> y(10);
  y <<
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

  const int n_vars   = x.cols();
  const float lambda = 0.1;
  const int seed     = 0;

  Forest<float, int> result = Forest<float, int>::train(
    TrainingSpecUGLDA<float, int>(n_vars, lambda),
    x,
    y,
    4,
    seed);


  Forest<float, int> expect;

  expect.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      TreeCondition<float, int>::make(
        as_projector({ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }),
        2.5,
        TreeResponse<float, int>::make(0),
        TreeResponse<float, int>::make(1)
        ))
    );

  expect.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      TreeCondition<float, int>::make(
        as_projector({ 0.9637967944145203, -0.02586924470961094, 0.10231951624155045, -0.1402283012866974,
                       -0.07096929848194122, -0.0709693655371666, -0.07096933573484421, -0.070969358086586,
                       -0.070969358086586, -0.0709693506360054, -0.07096933573484421, -0.07096933573484421 }),
        1.499821662902832,
        TreeResponse<float, int>::make(0),
        TreeResponse<float, int>::make(1)
        ))
    );

  expect.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      TreeCondition<float, int>::make(
        as_projector({ 1.0, -1.1535910715565478e-07, -2.1619270285100356e-07, 1.0954755680359085e-07,
                       6.949329645067337e-08, 6.949331066152808e-08, 6.949329645067337e-08, 6.949331066152808e-08,
                       6.949331066152808e-08, 6.949331066152808e-08, 6.949331066152808e-08, 6.949331066152808e-08 }),
        2.500000238418579,
        TreeResponse<float, int>::make(0),
        TreeResponse<float, int>::make(1)
        ))
    );

  expect.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      TreeCondition<float, int>::make(
        as_projector({ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }),
        2.5,
        TreeResponse<float, int>::make(0),
        TreeResponse<float, int>::make(1)
        ))
    );

  ASSERT_EQ(expect, result);
  ASSERT_EQ(expect.seed, result.seed);
}

TEST(Forest, PredictDataColumnSomeVariablesMultivariateThreeGroups) {
  Forest<float, int> forest;

  forest.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      TreeCondition<float, int>::make(
        as_projector({ 0.0, 0.0, 0.0, 0.5982325379690726, -0.8013225508589422 }),
        -0.3483987096124312,
        TreeCondition<float, int>::make(
          as_projector({ 0.999534397402818, 0.0, -0.030512102657559676, 0.0, 0.0 }),
          5.55339996020167,
          TreeResponse<float, int>::make(1),
          TreeResponse<float, int>::make(2)),
        TreeResponse<float, int>::make(0)
        ))
    );


  forest.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      TreeCondition<float, int>::make(
        as_projector({ 0.9998222455113714, 0.0, -0.018854107791118468, 0.0, 0.0 }),
        5.300417766337716,
        TreeCondition<float, int>::make(
          as_projector({ 0.9989543519613864, 0.0, 0.0, 0.0457187346435417, 0.0 }),
          1.6094899541803496,
          TreeResponse<float, int>::make(0),
          TreeResponse<float, int>::make(1)),
        TreeResponse<float, int>::make(2)))
    );


  forest.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      TreeCondition<float, int>::make(
        as_projector({ 0.9741975531020036, -0.2256969816591904, 0.0, 0.0, 0.0 }),
        3.9550147456664178,
        TreeCondition<float, int>::make(
          as_projector({ 0.0, 0.9995561292785718, -0.029791683766431428, 0.0, 0.0 }),
          2.6217629631670403,
          TreeResponse<float, int>::make(0),
          TreeResponse<float, int>::make(1)),
        TreeResponse<float, int>::make(2)))
    );


  forest.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      TreeCondition<float, int>::make(
        as_projector({ 0.9615748657636985, 0.0, 0.0, 0.0, -0.2745428519038971 }),
        4.734758305714628,
        TreeCondition<float, int>::make(
          as_projector({ 0.0, 0.0, 0.3772334858435029, 0.0, -0.926118187467647 }),
          -0.8315603229605784,
          TreeResponse<float, int>::make(1),
          TreeResponse<float, int>::make(0)),
        TreeResponse<float, int>::make(2)))
    );

  DataColumn<float> x(5);
  x << 9, 8, 1, 1, 1;

  int result = forest.predict(x);

  ASSERT_EQ(2, result);
}

TEST(Forest, PredictDataSomeVariablesMultivariateThreeGroups) {
  Forest<float, int> forest;

  forest.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      TreeCondition<float, int>::make(
        as_projector({ 0.0, 0.0, 0.0, 0.5982325379690726, -0.8013225508589422 }),
        -0.3483987096124312,
        TreeCondition<float, int>::make(
          as_projector({ 0.999534397402818, 0.0, -0.030512102657559676, 0.0, 0.0 }),
          5.55339996020167,
          TreeResponse<float, int>::make(1),
          TreeResponse<float, int>::make(2)),
        TreeResponse<float, int>::make(0))
      )
    );


  forest.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      TreeCondition<float, int>::make(
        as_projector({ 0.9998222455113714, 0.0, -0.018854107791118468, 0.0, 0.0 }),
        5.300417766337716,
        TreeCondition<float, int>::make(
          as_projector({ 0.9989543519613864, 0.0, 0.0, 0.0457187346435417, 0.0 }),
          1.6094899541803496,
          TreeResponse<float, int>::make(0),
          TreeResponse<float, int>::make(1)),
        TreeResponse<float, int>::make(2))
      )
    );


  forest.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      TreeCondition<float, int>::make(
        as_projector({ 0.9741975531020036, -0.2256969816591904, 0.0, 0.0, 0.0 }),
        3.9550147456664178,
        TreeCondition<float, int>::make(
          as_projector({ 0.0, 0.9995561292785718, -0.029791683766431428, 0.0, 0.0 }),
          2.6217629631670403,
          TreeResponse<float, int>::make(0),
          TreeResponse<float, int>::make(1)),
        TreeResponse<float, int>::make(2))
      )
    );


  forest.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      TreeCondition<float, int>::make(
        as_projector({ 0.9615748657636985, 0.0, 0.0, 0.0, -0.2745428519038971 }),
        4.734758305714628,
        TreeCondition<float, int>::make(
          as_projector({ 0.0, 0.0, 0.3772334858435029, 0.0, -0.926118187467647 }),
          -0.8315603229605784,
          TreeResponse<float, int>::make(1),
          TreeResponse<float, int>::make(0)),
        TreeResponse<float, int>::make(2))
      )
    );

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

  DataColumn<int> result = forest.predict(x);

  ASSERT_EQ(y.size(), result.size());
  ASSERT_EQ(y.cols(), result.cols());
  ASSERT_EQ(y.rows(), result.rows());
  ASSERT_EQ(y, result);
}
