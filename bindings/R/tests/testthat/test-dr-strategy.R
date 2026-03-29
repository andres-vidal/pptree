describe("dr_uniform", {
  it("creates a dr_strategy object with n_vars", {
    s <- dr_uniform(n_vars = 2)
    expect_s3_class(s, "dr_strategy")
    expect_equal(s$name, "uniform")
    expect_equal(s$n_vars, 2)
    expect_null(s$p_vars)
  })

  it("creates a dr_strategy object with p_vars", {
    s <- dr_uniform(p_vars = 0.5)
    expect_s3_class(s, "dr_strategy")
    expect_equal(s$name, "uniform")
    expect_null(s$n_vars)
    expect_equal(s$p_vars, 0.5)
  })

  it("creates a dr_strategy object with p_vars=1", {
    s <- dr_uniform(p_vars = 1)
    expect_s3_class(s, "dr_strategy")
    expect_equal(s$name, "uniform")
    expect_null(s$n_vars)
    expect_equal(s$p_vars, 1)
  })

  it("rejects both n_vars and p_vars", {
    expect_error(dr_uniform(n_vars = 2, p_vars = 0.5), "not both")
  })

  it("rejects invalid n_vars", {
    expect_error(dr_uniform(n_vars = 0), "positive integer greater than 0")
    expect_error(dr_uniform(n_vars = 1.5), "positive integer greater than 0")
  })

  it("rejects invalid p_vars", {
    expect_error(dr_uniform(p_vars = 0), "between 0 (exclusive) and 1 (inclusive)", fixed = TRUE)
    expect_error(dr_uniform(p_vars = 2), "between 0 (exclusive) and 1 (inclusive)", fixed = TRUE)
  })
})

describe("dr_noop", {
  it("creates a dr_strategy object", {
    s <- dr_noop()
    expect_s3_class(s, "dr_strategy")
    expect_equal(s$name, "noop")
  })
})
