test_that('a bare nn_conditional throws an error when called', {
  conditional <- nn_conditional()
  expect_error(conditional(torch_randn(10), torch_randn(5)))
})

test_that('nn_conditional_mlp initializes correctly', {
  mlp <- nn_conditional_mlp(10, 5, 1)
  expect_true(!is.null(mlp$model))
})

test_that('nn_conditional_mlp forward pass works', {
  mlp <- nn_conditional_mlp(10, 5, 1)
  input <- torch_randn(10)
  conditioning <- torch_randn(5)
  output <- mlp(input, conditioning)
  
  expect_equal(output$size(), 1)
})

test_that('nn_conditional_mlp forward pass works with batch dimensions', {
  mlp <- nn_conditional_mlp(10, 5, 1)
  n_batch <- 8
  input <- torch_randn(n_batch, 10)
  conditioning <- torch_randn(n_batch, 5)
  output <- mlp(input, conditioning)
  
  expect_equal(output$size(), c(n_batch, 1))
})

test_that('nn_conditional_mlp handles missing conditioning', {
  mlp <- nn_conditional_mlp(10, 0, 1)
  input <- torch_randn(10)
  output <- mlp(input)
  
  expect_equal(output$size(), 1)
})

test_that('nn_conditional_mlp handles different layer sizes', {
  mlp <- nn_conditional_mlp(10, 5, 1, layer_sizes = c(64, 32))
  input <- torch_randn(10)
  conditioning <- torch_randn(5)
  output <- mlp(input, conditioning)
  
  expect_equal(output$size(), 1)
})

test_that('nn_conditional_mlp allows alternative activation functions', {
  mlp <- nn_conditional_mlp(10, 5, 1, activation = nn_tanh)
  input <- torch_randn(10)
  conditioning <- torch_randn(5)
  output <- mlp(input, conditioning)
  
  expect_equal(output$size(), 1)
})
