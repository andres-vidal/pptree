describe("pp_pda", {
  it("creates a pp_strategy object", {
    s <- pp_pda(0.5)
    expect_s3_class(s, "pp_strategy")
    expect_equal(s$name, "pda")
    expect_equal(s$lambda, 0.5)
  })

  it("defaults lambda to 0", {
    s <- pp_pda()
    expect_equal(s$lambda, 0)
  })

  it("rejects invalid lambda", {
    expect_error(pp_pda(-1), "between 0 and 1")
    expect_error(pp_pda(2), "between 0 and 1")
    expect_error(pp_pda("a"), "between 0 and 1")
  })
})
