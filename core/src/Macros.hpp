
#define APPROX_THRESHOLD 0.01

#define ASSERT_APPROX(a, b)    ASSERT_TRUE(a.isApprox(b, APPROX_THRESHOLD)) << "Expected " << std::endl << a << std::endl << " to be approximate to " << std::endl << b
#define ASSERT_COLLINEAR(a, b) ASSERT_TRUE(models::math::collinear(a, b)) << "Expected columns of " << std::endl << a << std::endl << " to be collinear with its respective column of " << std::endl << b


#define ASSERT_EQ_MATRIX(a, b) \
        ASSERT_EQ(a.size(), b.size()); \
        ASSERT_EQ(a.rows(), b.rows()); \
        ASSERT_EQ(a.cols(), b.cols()); \
        ASSERT_EQ(a, b);
