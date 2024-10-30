test_that('nn_sequential_conditional_flow outputs have the correct structure', {
  input_size <- 10
  conditioning_size <- 5
  
  flow_model <- nn_sequential_conditional_flow(
    nn_affine_coupling_block(input_size, conditioning_size),
    nn_permutation_flow(input_size),
    nn_affine_coupling_block(input_size, conditioning_size)
  )
  
  input <- torch_randn(input_size)
  conditioning <- torch_randn(conditioning_size)
  output <- flow_model(input, conditioning)
  
  expect_equal(output$size(), input$size())
  expect_equal(attr(output, 'log_jacobian')$size(), 1)
})

test_that('nn_sequential_conditional_flow reverse pass restores input', {
  input_size <- 10
  conditioning_size <- 5
  
  flow_model <- nn_sequential_conditional_flow(
    nn_affine_coupling_block(input_size, conditioning_size),
    nn_permutation_flow(input_size),
    nn_affine_coupling_block(input_size, conditioning_size)
  )
  
  input <- torch_randn(10, input_size)
  conditioning <- torch_randn(10, conditioning_size)
  output <- flow_model(input, conditioning)
  restored_input <- flow_model$reverse(output, conditioning)
  
  expect_equal(as_array(restored_input), as_array(input), tolerance = 1e-5)
})

test_that('nn_sequential_conditional_flow handles batch dimensions', {
  input_size <- 10
  conditioning_size <- 5
  n_batch <- 5
  
  flow_model <- nn_sequential_conditional_flow(
    nn_affine_coupling_block(input_size, conditioning_size),
    nn_permutation_flow(input_size),
    nn_affine_coupling_block(input_size, conditioning_size)
  )
  
  input <- torch_randn(n_batch, input_size)
  conditioning <- torch_randn(n_batch, conditioning_size)
  output <- flow_model(input, conditioning)
  
  expect_equal(output$size(), c(n_batch, input_size))
  expect_equal(attr(output, 'log_jacobian')$size(), c(n_batch, 1))
})

test_that('nn_sequential_conditional_flow handles missing conditioning in forward and reverse pass', {
  input_size <- 10
  flow_model <- nn_sequential_conditional_flow(
    nn_affine_coupling_block(input_size)
  )
  
  input <- torch_randn(input_size)
  output <- flow_model(input)
  
  expect_equal(output$size(), input$size())
  expect_equal(attr(output, 'log_jacobian')$size(), 1)
  
  restored_input <- flow_model$reverse(output)
  expect_equal(as_array(restored_input), as_array(input), tolerance = 1e-5)
})

test_that('nn_sequential_conditional_flow dimension method returns correct size', {
  input_size <- 10
  conditioning_size <- 5
  
  flow_model <- nn_sequential_conditional_flow(
    nn_affine_coupling_block(input_size, conditioning_size),
    nn_permutation_flow(input_size),
    nn_affine_coupling_block(input_size, conditioning_size)
  )
  
  expect_equal(flow_model$dimension(), input_size)
})

custom_constant_jacobian_flow <- nn_module(
  inherit = nn_conditional_flow,
  forward = function(input, conditioning) {
    attr(input, 'log_jacobian') <- torch_ones(input$size(1), 1, device = input$device)
    input
  },
  reverse = function(input, conditioning) {
    input
  }
)

test_that('nn_sequential_conditional_flow propagates log_jacobian correctly', {
  input_size <- 10
  conditioning_size <- 5
  
  # Create a sequential flow model with custom flows
  flow_model <- nn_sequential_conditional_flow(
    custom_constant_jacobian_flow(),
    custom_constant_jacobian_flow(),
    custom_constant_jacobian_flow()
  )
  
  input <- torch_randn(10, input_size)
  conditioning <- torch_randn(10, conditioning_size)
  output <- flow_model(input, conditioning)
  
  # The expected log_jacobian should be 3 since each flow adds 1
  expected_log_jacobian <- torch_full(c(10, 1), 3, device = input$device)
  
  expect_equal(as_array(attr(output, 'log_jacobian')), as_array(expected_log_jacobian))
})
