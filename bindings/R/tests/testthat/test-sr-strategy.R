describe("sr_mean_of_means", {
  it("creates a sr_strategy object", {
    s <- sr_mean_of_means()
    expect_s3_class(s, "sr_strategy")
    expect_equal(s$name, "mean_of_means")
  })
})
