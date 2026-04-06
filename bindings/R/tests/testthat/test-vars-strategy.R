describe("vars_uniform", {
  it("creates a vars_strategy object with n_vars", {
    s <- vars_uniform(n_vars = 2)
    expect_s3_class(s, "vars_strategy")
    expect_equal(s$name, "uniform")
    expect_equal(s$count, 2)
    expect_null(s$p_vars)
  })

  it("creates a vars_strategy object with p_vars", {
    s <- vars_uniform(p_vars = 0.5)
    expect_s3_class(s, "vars_strategy")
    expect_equal(s$name, "uniform")
    expect_null(s$count)
    expect_equal(s$p_vars, 0.5)
  })

  it("creates a vars_strategy object with p_vars=1", {
    s <- vars_uniform(p_vars = 1)
    expect_s3_class(s, "vars_strategy")
    expect_equal(s$name, "uniform")
    expect_null(s$count)
    expect_equal(s$p_vars, 1)
  })

  it("rejects both n_vars and p_vars", {
    expect_error(vars_uniform(n_vars = 2, p_vars = 0.5), "not both")
  })

  it("rejects invalid n_vars", {
    expect_error(vars_uniform(n_vars = 0), "positive integer greater than 0")
    expect_error(vars_uniform(n_vars = 1.5), "positive integer greater than 0")
  })

  it("rejects invalid p_vars", {
    expect_error(vars_uniform(p_vars = 0), "between 0 (exclusive) and 1 (inclusive)", fixed = TRUE)
    expect_error(vars_uniform(p_vars = 2), "between 0 (exclusive) and 1 (inclusive)", fixed = TRUE)
  })
})

describe("vars_all", {
  it("creates a vars_strategy object", {
    s <- vars_all()
    expect_s3_class(s, "vars_strategy")
    expect_equal(s$name, "all")
  })
})
