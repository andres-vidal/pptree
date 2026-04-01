/**
 * @file Normal.test.cpp
 * @brief Statistical correctness tests for the Normal (Gaussian) random
 *        number generator.
 *
 * These tests validate that the Box-Muller transform in Normal produces
 * outputs whose distributional properties match the theoretical N(μ, σ²).
 * Every test uses a fixed seed, making results fully deterministic and
 * reproducible across platforms.
 *
 * The test suite is organised into four groups:
 *
 *   1. **Moment tests** — verify that the first four central moments
 *      (mean, variance, skewness, kurtosis) converge to their theoretical
 *      values for large N.  The normal distribution has skewness = 0 and
 *      (non-excess) kurtosis = 3; deviations from these indicate a broken
 *      transform or incorrect uniform inputs.
 *
 *   2. **Kolmogorov-Smirnov goodness-of-fit** — the gold-standard
 *      non-parametric test comparing the empirical CDF to the theoretical
 *      Φ(x).  The K-S test is sensitive to any departure from normality
 *      (location, scale, shape, or tail behaviour), making it a strong
 *      end-to-end check of both gen_unif01() and the Box-Muller transform.
 *
 *   3. **Lag-1 autocorrelation** — Box-Muller produces pairs (Z1, Z2)
 *      from the same (U1, U2), so consecutive outputs could be correlated
 *      if the caching logic is incorrect or if (Z1, Z2) are not truly
 *      independent conditioned on the uniform inputs.  We verify the
 *      lag-1 sample autocorrelation is near zero.
 *
 *   4. **Batch generation** — the multi-sample overload should produce the
 *      same distributional properties as repeated single draws.
 *
 * ## Why fixed seeds make these tests deterministic
 *
 * Statistical tests are inherently probabilistic — a correct generator
 * can produce a sample that exceeds the critical value by chance.  By
 * fixing the seed, the sample is the same on every run, so each test
 * statistic is a constant.  If it passes once, it passes always.  If
 * the implementation changes and the constant crosses the critical
 * threshold, the test fails, signalling a distributional regression.
 *
 * ## Tolerance derivation
 *
 * For N i.i.d. draws from N(μ, σ²), the CLT gives standard errors for
 * each sample moment:
 *
 *   SE(mean)     = σ / √N
 *   SE(variance) ≈ σ² √(2 / (N−1))
 *   SE(skewness) ≈ √(6 / N)
 *   SE(kurtosis) ≈ √(24 / N)
 *
 * Tolerances are computed as exactly 5 SE in each test body, wide enough
 * that a correct generator under any fixed seed passes comfortably,
 * while a broken generator (e.g. one that repeats cached values) fails
 * by orders of magnitude.
 *
 * ## Kolmogorov-Smirnov critical values
 *
 * The two-sided K-S statistic D_N = sup_x |F_N(x) − F(x)| has the
 * asymptotic distribution P(√N · D_N ≤ t) → K(t) (the Kolmogorov
 * distribution).  For significance level α, the critical value is:
 *
 *   D_crit = c(α) / √N
 *
 * where c(0.001) = 1.9495.  With N = 100 000 this gives D_crit ≈ 0.0062.
 *
 * @see Box & Muller, "A Note on the Generation of Random Normal Deviates"
 *      (Annals of Math. Stat., 1958).
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "stats/Normal.hpp"

using namespace ppforest2::stats;

/**
 * @brief Standard normal CDF: Φ(x) = ½ erfc(−x / √2).
 *
 * Used by the Kolmogorov-Smirnov tests to compute the theoretical CDF
 * at each order statistic.  Delegates to std::erfc which is required by
 * C++11 and provides full double-precision accuracy.
 */
static double phi(double x) {
  return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

// ---------------------------------------------------------------------------
// Moment tests
//
// For X ~ N(μ, σ²):
//   E[X]       = μ
//   Var[X]     = σ²
//   Skew[X]    = 0          (symmetric distribution)
//   Kurt[X]    = 3          (non-excess kurtosis; excess = 0)
//
// Each test draws N = 100 000 samples and checks convergence within a
// tolerance derived from the CLT.
// ---------------------------------------------------------------------------

/**
 * Draws from N(5, 4) and verifies the sample mean converges to 5.
 *
 * Uses μ = 5, σ = 2 to test a non-zero-centred distribution, ensuring
 * the denormalize() step (mean + std_dev * z) is correct.
 */
TEST(Normal, MeanConvergesToTheory) {
  RNG rng(0);
  Normal normal(5.0f, 2.0f);

  constexpr int N     = 100000;
  constexpr double MU = 5.0;
  constexpr double SD = 2.0;

  double sum = 0;

  for (int i = 0; i < N; i++) {
    sum += normal(rng);
  }

  double sample_mean = sum / N;
  // SE(mean) = σ / √N
  double se = SD / std::sqrt(static_cast<double>(N));
  ASSERT_NEAR(sample_mean, MU, 5 * se);
}

/**
 * Draws from N(0, 9) and verifies the sample variance converges to 9.
 *
 * Uses μ = 0, σ = 3. The sample variance (with Bessel correction, N−1
 * denominator) is an unbiased estimator of σ².
 */
TEST(Normal, VarianceConvergesToTheory) {
  RNG rng(0);
  Normal normal(0.0f, 3.0f);

  constexpr int N           = 100000;
  constexpr double SIGMA_SQ = 9.0; // σ² = 3² = 9

  std::vector<double> samples(N);

  for (int i = 0; i < N; i++) {
    samples[i] = normal(rng);
  }

  double mean   = std::accumulate(samples.begin(), samples.end(), 0.0) / N;
  double sum_sq = 0;

  for (double x : samples) {
    sum_sq += (x - mean) * (x - mean);
  }

  double sample_var = sum_sq / (N - 1);
  // SE(s²) ≈ σ² √(2/(N−1))
  double se = SIGMA_SQ * std::sqrt(2.0 / (N - 1));
  ASSERT_NEAR(sample_var, SIGMA_SQ, 5 * se);
}

/**
 * Draws from N(0, 1) and verifies skewness ≈ 0.
 *
 * The sample skewness is computed as  γ₁ = m₃ / m₂^{3/2}  where
 * m_k = (1/N) Σ (x_i − x̄)^k  are the k-th central sample moments.
 * For a symmetric distribution, γ₁ = 0.
 */
TEST(Normal, SkewnessNearZero) {
  RNG rng(0);
  Normal normal(0.0f, 1.0f);

  constexpr int N = 100000;
  std::vector<double> samples(N);

  for (int i = 0; i < N; i++) {
    samples[i] = normal(rng);
  }

  double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / N;
  double m2 = 0, m3 = 0;

  for (double x : samples) {
    double d = x - mean;
    m2 += d * d;
    m3 += d * d * d;
  }

  m2 /= N;
  m3 /= N;

  double skewness = m3 / std::pow(m2, 1.5);
  // SE(γ₁) ≈ √(6/N)
  double se = std::sqrt(6.0 / N);
  ASSERT_NEAR(skewness, 0.0, 5 * se);
}

/**
 * Draws from N(0, 1) and verifies (non-excess) kurtosis ≈ 3.
 *
 * The sample kurtosis is  κ = m₄ / m₂²  where m_k are central moments.
 * The normal distribution has κ = 3; heavier or lighter tails shift this
 * value.  This test catches tail problems in gen_unif01() (e.g. if the
 * 53-bit precision were reduced, the extreme tails would be truncated,
 * lowering the kurtosis).
 */
TEST(Normal, KurtosisNearThree) {
  RNG rng(0);
  Normal normal(0.0f, 1.0f);

  constexpr int N = 100000;
  std::vector<double> samples(N);

  for (int i = 0; i < N; i++) {
    samples[i] = normal(rng);
  }

  double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / N;
  double m2 = 0, m4 = 0;

  for (double x : samples) {
    double d = x - mean;
    m2 += d * d;
    m4 += d * d * d * d;
  }

  m2 /= N;
  m4 /= N;

  double kurtosis = m4 / (m2 * m2);
  // SE(κ) ≈ √(24/N)
  double se = std::sqrt(24.0 / N);
  ASSERT_NEAR(kurtosis, 3.0, 5 * se);
}

// ---------------------------------------------------------------------------
// Kolmogorov-Smirnov goodness-of-fit
//
// The two-sided K-S test compares the empirical CDF F_N(x) against the
// theoretical CDF F(x) = Φ((x − μ) / σ):
//
//   D_N = sup_x |F_N(x) − F(x)|
//
// The empirical CDF is a step function that jumps by 1/N at each order
// statistic x_(i).  Between jumps the supremum occurs at either the
// left or right edge of the step, giving:
//
//   D_N = max_i { |F(x_(i)) − (i−1)/N|,  |F(x_(i)) − i/N| }
//
// (using 0-based indexing: i/N and (i+1)/N).
//
// Under H₀ (samples truly from F), √N · D_N converges in distribution
// to the Kolmogorov distribution K(t).  The critical value at
// significance level α is  D_crit = c(α) / √N, where:
//
//   c(0.10)  = 1.2238
//   c(0.05)  = 1.3581
//   c(0.01)  = 1.6276
//   c(0.001) = 1.9495
//
// We use α = 0.001 for a very conservative threshold.  With N = 100 000:
//   D_crit = 1.9495 / √100000 ≈ 0.00616
//
// This is sensitive enough to detect location shifts of ~0.02σ, scale
// errors of ~1%, and shape distortions in the tails.
// ---------------------------------------------------------------------------

/**
 * K-S test against the standard normal Φ(x).
 *
 * Tests the full pipeline: gen_unif01() → Box-Muller → caching → output.
 */
TEST(Normal, KolmogorovSmirnovStandardNormal) {
  RNG rng(0);
  Normal normal(0.0f, 1.0f);

  constexpr int N = 100000;
  std::vector<double> samples(N);

  for (int i = 0; i < N; i++) {
    samples[i] = normal(rng);
  }

  std::sort(samples.begin(), samples.end());

  double d_max = 0;

  for (int i = 0; i < N; i++) {
    double F  = phi(samples[i]);
    double d1 = std::abs(F - static_cast<double>(i) / N);
    double d2 = std::abs(F - static_cast<double>(i + 1) / N);
    d_max     = std::max(d_max, std::max(d1, d2));
  }

  // α = 0.001,  D_crit = 1.9495 / √N ≈ 0.00616
  double d_crit = 1.9495 / std::sqrt(static_cast<double>(N));
  ASSERT_LT(d_max, d_crit);
}

/**
 * K-S test for N(10, 25) — a shifted and scaled distribution.
 *
 * The samples are standardised before computing Φ: z = (x − μ) / σ.
 * This tests that denormalize() correctly applies the affine transform,
 * and uses a different seed (99) for variety.
 */
TEST(Normal, KolmogorovSmirnovShiftedNormal) {
  RNG rng(99);
  Normal normal(10.0f, 5.0f);

  constexpr int N = 100000;
  std::vector<double> samples(N);

  for (int i = 0; i < N; i++) {
    samples[i] = normal(rng);
  }

  std::sort(samples.begin(), samples.end());

  double d_max = 0;

  for (int i = 0; i < N; i++) {
    double F  = phi((samples[i] - 10.0) / 5.0);
    double d1 = std::abs(F - static_cast<double>(i) / N);
    double d2 = std::abs(F - static_cast<double>(i + 1) / N);
    d_max     = std::max(d_max, std::max(d1, d2));
  }

  double d_crit = 1.9495 / std::sqrt(static_cast<double>(N));
  ASSERT_LT(d_max, d_crit);
}

// ---------------------------------------------------------------------------
// Independence — lag-1 autocorrelation
//
// Box-Muller generates pairs (Z1, Z2) from the same (U1, U2).  The
// caching logic returns Z1 on one call and Z2 on the next, so
// consecutive outputs alternate between freshly-generated and cached
// values.  If the caching is broken (e.g. the cached value is never
// consumed, or the same U1/U2 are reused), consecutive samples will be
// correlated.
//
// The lag-1 sample autocorrelation is:
//
//   r₁ = [Σ_{i=2}^{N} (x_i − x̄)(x_{i-1} − x̄)]  /  [Σ_{i=1}^{N} (x_i − x̄)²]
//
// For i.i.d. samples, r₁ → 0.  Its standard error is approximately
// 1/√N.
// ---------------------------------------------------------------------------

TEST(Normal, Lag1AutocorrelationNearZero) {
  RNG rng(0);
  Normal normal(0.0f, 1.0f);

  constexpr int N = 100000;
  std::vector<double> samples(N);

  for (int i = 0; i < N; i++) {
    samples[i] = normal(rng);
  }

  double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / N;

  double num = 0, den = 0;

  for (int i = 0; i < N; i++) {
    double d = samples[i] - mean;
    den += d * d;

    if (i > 0) {
      num += d * (samples[i - 1] - mean);
    }
  }

  double r1 = num / den;
  // SE(r₁) ≈ 1/√N
  double se = 1.0 / std::sqrt(static_cast<double>(N));
  ASSERT_NEAR(r1, 0.0, 5 * se);
}

// ---------------------------------------------------------------------------
// Batch generation
//
// The multi-sample overload operator()(count, rng) should produce the
// same distributional properties as repeated single draws, and return a
// vector of exactly the requested size.
// ---------------------------------------------------------------------------

TEST(Normal, BatchMeanConverges) {
  RNG rng(0);
  Normal normal(0.0f, 1.0f);

  auto samples = normal(100000, rng);

  double sum = 0;

  for (float x : samples) {
    sum += x;
  }

  double sample_mean = sum / samples.size();
  // σ = 1,  SE(mean) = σ / √N
  double se = 1.0 / std::sqrt(static_cast<double>(samples.size()));
  ASSERT_NEAR(sample_mean, 0.0, 5 * se);
}

TEST(Normal, BatchSizeIsCorrect) {
  RNG rng(0);
  Normal normal(0.0f, 1.0f);

  auto samples = normal(500, rng);
  ASSERT_EQ(samples.size(), 500u);
}
