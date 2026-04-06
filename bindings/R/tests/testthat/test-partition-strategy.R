describe("partition_by_group", {
  it("creates a partition_strategy object", {
    s <- partition_by_group()
    expect_s3_class(s, "partition_strategy")
    expect_equal(s$name, "by_group")
  })
})
