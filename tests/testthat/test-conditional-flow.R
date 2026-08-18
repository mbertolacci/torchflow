test_that('nn_conditional_flow forward method raises error', {
  conditional_flow <- nn_conditional_flow()
  input <- torch_randn(10, 2)
  conditioning <- torch_randn(10, 2)
  
  expect_error(conditional_flow$forward(input, conditioning), 'Not implemented')
})

test_that('nn_conditional_flow provides a luz-compatible loss method', {
  conditional_flow <- nn_conditional_flow()

  expect_identical(names(formals(conditional_flow$loss)), c('input', 'target'))
})

test_that('nn_conditional_flow can be set up with luz without an explicit loss', {
  skip_if_not_installed('luz')

  expect_no_error(
    luz::setup(nn_conditional_flow, optimizer = torch::optim_adam)
  )
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
