test_that('nn_permutation_flow forward pass permutes input', {
  permutation_flow <- nn_permutation_flow(10)
  input <- torch_arange(1, 10)
  output <- permutation_flow(input)
  
  expect_equal(sort(as_array(output)), as_array(input))
  expect_equal(attr(output, 'log_jacobian'), torch_zeros(1))
})

test_that('nn_permutation_flow reverse pass restores input', {
  permutation_flow <- nn_permutation_flow(10)
  input <- torch_arange(1, 10)
  output <- permutation_flow(input)
  restored_input <- permutation_flow$reverse(output)
  
  expect_equal(as_array(restored_input), as_array(input))
})

test_that('nn_permutation_flow handles batch dimensions', {
  permutation_flow <- nn_permutation_flow(10)
  n_batch <- 5
  input <- torch_arange(1, 10)$unsqueeze(1)$expand(c(n_batch, -1))
  output <- permutation_flow(input)
  
  expect_equal(output$size(), c(n_batch, 10))
  expect_equal(attr(output, 'log_jacobian')$size(), c(n_batch, 1))
})

test_that('nn_permutation_flow dimension method returns correct size', {
  permutation_flow <- nn_permutation_flow(10)
  expect_equal(permutation_flow$dimension(), 10)
})
