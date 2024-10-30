test_that('nn_conditional_flow forward method raises error', {
  conditional_flow <- nn_conditional_flow()
  input <- torch_randn(10, 2)
  conditioning <- torch_randn(10, 2)
  
  expect_error(conditional_flow$forward(input, conditioning), 'Not implemented')
})

test_that('nn_conditional_flow reverse method raises error', {
  conditional_flow <- nn_conditional_flow()
  input <- torch_randn(10, 2)
  conditioning <- torch_randn(10, 2)
  
  expect_error(conditional_flow$reverse(input, conditioning), 'Not implemented')
})

test_that('nn_conditional_flow dimension method raises error', {
  conditional_flow <- nn_conditional_flow()
  
  expect_error(conditional_flow$dimension(), 'Not implemented')
})
