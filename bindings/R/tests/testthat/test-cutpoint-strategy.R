describe("cutpoint_mean_of_means", {
  it("creates a cutpoint_strategy object", {
    s <- cutpoint_mean_of_means()
    expect_s3_class(s, "cutpoint_strategy")
    expect_equal(s$name, "mean_of_means")
  })
})
