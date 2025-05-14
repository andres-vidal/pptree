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
        as_projector({ 0.9580563306808472, -0.1769358515739441, 0.006788997910916805, 0.10231061279773712, -0.2007693350315094 }),
        3.9684371948242188,
        TreeCondition<float, int>::make(
          as_projector({ 0.037195101380348206, 0.9892118573188782, -0.038433052599430084, 0.03852792829275131, 0.13082443177700043 }),
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
        as_projector({ 0.9805806875228882, -0.19611600041389465, -7.279736990994934e-08, 1.1245992226349699e-07, -9.458754135494019e-08 }),
        4.118439674377441,
        TreeCondition<float, int>::make(
          as_projector({ 0.0, 1.0, -1.9481838364754367e-08, 1.0491407920198981e-07, -4.4338150928524556e-08 }),
          2.5,
          TreeResponse<float, int>::make(0),
          TreeResponse<float, int>::make(1)
          ),
        TreeResponse<float, int>::make(2)
        ))
    );

  expect.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      TreeCondition<float, int>::make(
        as_projector({ 0.9729747772216797, -0.1960887610912323, -0.038586050271987915, 0.0533757247030735, -0.10262320935726166 }),
        3.9661753177642822,
        TreeCondition<float, int>::make(
          as_projector({ 0.2779413163661957, 0.9268009662628174, -0.11087987571954727, 0.06655726581811905, 0.2169434279203415 }),
          3.0928874015808105,
          TreeResponse<float, int>::make(0),
          TreeResponse<float, int>::make(1)
          ),
        TreeResponse<float, int>::make(2)
        ))
    );

  expect.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      TreeCondition<float, int>::make(
        as_projector({ 0.9740977883338928, -0.19465802609920502, -0.022930234670639038, 0.08263365924358368, -0.07673182338476181 }),
        4.049604415893555,
        TreeCondition<float, int>::make(
          as_projector({ 0.14019571244716644, 0.9772301912307739, 0.0012194644659757614, 0.025386467576026917, 0.157227024435997 }),
          2.9420154094696045,
          TreeResponse<float, int>::make(0),
          TreeResponse<float, int>::make(1)
          ),
        TreeResponse<float, int>::make(2)
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
        as_projector({ 0.9023475646972656, 0.0, 0.0, 0.4310089647769928, 0.0 }),
        5.072210311889648,
        TreeCondition<float, int>::make(
          as_projector({ 0.0, 0.9889134764671326, 0.0, 0.14849309623241425, 0.0 }),
          2.7555019855499268,
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
        as_projector({ 0.9626245498657227, 0.0, 0.0, 0.0, -0.2708394229412079 }),
        4.723257541656494,
        TreeCondition<float, int>::make(
          as_projector({ 0.0, 1.0, 0.0, 0.0, 0.0 }),
          2.5,
          TreeResponse<float, int>::make(0),
          TreeResponse<float, int>::make(1)
          ),
        TreeResponse<float, int>::make(2)
        ))
    );

  // Third tree
  expect.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      TreeCondition<float, int>::make(
        as_projector({ 0.0, 0.0, 0.43462103605270386, 0.0, 0.9006133675575256 }),
        1.1528732776641846,
        TreeResponse<float, int>::make(0),
        TreeCondition<float, int>::make(
          as_projector({ 0.0, 0.0, -0.9125092029571533, 0.0, 0.4090558588504791 }),
          0.10619720071554184,
          TreeResponse<float, int>::make(1),
          TreeResponse<float, int>::make(2)
          ))
      )
    );

  // Fourth tree
  expect.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      TreeCondition<float, int>::make(
        as_projector({ -0.9543675780296326, 0.0, 0.2986343502998352, 0.0, 0.0 }),
        -4.9326019287109375,
        TreeResponse<float, int>::make(2),
        TreeCondition<float, int>::make(
          as_projector({ 0.0, 0.9989996552467346, 0.0, 0.04471937566995621, 0.0 }),
          2.6413731575012207,
          TreeResponse<float, int>::make(0),
          TreeResponse<float, int>::make(1)
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
        as_projector({ 0.9404149055480957, 0.1384861171245575, 0.27520284056663513, -0.028697863221168518,
                       -0.04985111951828003, -0.04985113441944122, -0.04985114187002182, -0.04985113441944122,
                       -0.04985114559531212, -0.049851153045892715, -0.04985114559531212, -0.04985113441944122 }),
        2.056182384490967,
        TreeResponse<float, int>::make(0),
        TreeResponse<float, int>::make(1)
        ))
    );

  expect.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      TreeCondition<float, int>::make(
        as_projector({ 0.9504902362823486, -0.032086681574583054, -0.0, -0.0015316768549382687,
                       -0.10927978157997131, -0.10927978903055191, -0.10927974432706833, -0.10927967727184296,
                       -0.10927972197532654, -0.10927971452474594, -0.10927971452474594, -0.10927971452474594 }),
        1.0509742498397827,
        TreeResponse<float, int>::make(0),
        TreeResponse<float, int>::make(1)
        ))
    );

  expect.add_tree(
    std::make_unique<BootstrapTree<float, int> >(
      TreeCondition<float, int>::make(
        as_projector({ 1.0, -1.4527107339290524e-07, -2.9038861271146743e-07, -9.766825570522997e-08,
                       8.5481985934166e-08, 8.548196461788393e-08, 8.54819361961745e-08, 8.54819361961745e-08,
                       8.54819361961745e-08, 8.54819361961745e-08, 8.54819361961745e-08, 8.54819361961745e-08 }),
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
