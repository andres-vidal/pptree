describe("stop_pure_node", {
  it("creates a stop_strategy object", {
    s <- stop_pure_node()
    expect_s3_class(s, "stop_strategy")
    expect_equal(s$name, "pure_node")
  })
})
