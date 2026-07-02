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

test_that('forward_kl_loss handles univariate flow output', {
  flow_model <- suppressWarnings(nn_sequential_conditional_flow(
    nn_affine_coupling_block(1),
    nn_permutation_flow(1),
    nn_affine_coupling_block(1)
  ))

  input <- torch_randn(10, 1)
  output <- flow_model(input)
  loss <- forward_kl_loss(output)

  expect_true(is.finite(as_array(loss)))
})

test_that('forward_kl_loss handles spline flow output', {
  flow_model <- nn_sequential_conditional_flow(
    nn_dual_coupling_block(2, transform = "spline", bins = 4),
    nn_permutation_flow(2),
    nn_dual_coupling_block(2, transform = "spline", bins = 4)
  )

  input <- torch_randn(10, 2)
  output <- flow_model(input)
  loss <- forward_kl_loss(output)

  expect_true(is.finite(as_array(loss)))
})

test_that('forward_kl_loss handles actnorm flow output', {
  flow_model <- nn_sequential_conditional_flow(
    nn_actnorm_block(2),
    nn_spline_coupling_block(2, bins = 4),
    nn_permutation_flow(2),
    nn_actnorm_block(2),
    nn_spline_coupling_block(2, bins = 4)
  )

  input <- torch_randn(10, 2)
  output <- flow_model(input)
  loss <- forward_kl_loss(output)

  expect_true(is.finite(as_array(loss)))
})
