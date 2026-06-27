test_that('nn_spline_coupling_block forward pass transforms input correctly', {
  input_size <- 4
  conditioning_size <- 2
  coupling_block <- nn_spline_coupling_block(input_size, conditioning_size)
  input <- torch_randn(10, input_size)
  conditioning <- torch_randn(10, conditioning_size)

  output <- coupling_block(input, conditioning)

  expect_equal(output$size(), input$size())
  expect_equal(attr(output, 'log_jacobian')$size(), c(10, 1))
})

test_that('nn_spline_coupling_block reverse pass restores input', {
  input_size <- 4
  conditioning_size <- 2
  coupling_block <- nn_spline_coupling_block(input_size, conditioning_size)
  input <- torch_randn(10, input_size)
  conditioning <- torch_randn(10, conditioning_size)

  output <- coupling_block(input, conditioning)
  restored_input <- coupling_block$reverse(output, conditioning)

  expect_true(as.logical(torch_allclose(input, restored_input, atol = 1e-4)))
})

test_that('nn_spline_coupling_block handles missing conditioning', {
  coupling_block <- nn_spline_coupling_block(4)
  input <- torch_randn(10, 4)

  output <- coupling_block(input)
  restored_input <- coupling_block$reverse(output)

  expect_equal(output$size(), input$size())
  expect_true(as.logical(torch_allclose(input, restored_input, atol = 1e-4)))
})

test_that('nn_spline_coupling_block dimension method returns correct size', {
  coupling_block <- nn_spline_coupling_block(4)

  expect_equal(coupling_block$dimension(), 4)
})

test_that('nn_spline_coupling_block can be used in sequential conditional flow', {
  input_size <- 4
  conditioning_size <- 2
  flow_model <- nn_sequential_conditional_flow(
    nn_spline_coupling_block(input_size, conditioning_size),
    nn_permutation_flow(input_size),
    nn_spline_coupling_block(input_size, conditioning_size)
  )
  input <- torch_randn(10, input_size)
  conditioning <- torch_randn(10, conditioning_size)

  output <- flow_model(input, conditioning)
  restored_input <- flow_model$reverse(output, conditioning)

  expect_equal(output$size(), input$size())
  expect_true(as.logical(torch_allclose(input, restored_input, atol = 1e-4)))
})
