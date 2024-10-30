test_that('forward_kl_loss computes correct loss for standard_normal distribution', {
  input <- torch_ones(1, 2)
  attr(input, 'log_jacobian') <- torch_zeros(1)
  loss <- forward_kl_loss(input)
  
  expected_loss <- 0.5 * 2 - 0
  expect_equal(as_array(loss), expected_loss)
})

test_that('forward_kl_loss subtracts log_jacobian correctly', {
  input <- torch_zeros(10, 2)
  log_jacobian <- torch_rand(10)
  attr(input, 'log_jacobian') <- log_jacobian
  loss <- forward_kl_loss(input)
  
  expected_loss <- -torch_mean(log_jacobian)
  expect_equal(as_array(loss), as_array(expected_loss))
})
