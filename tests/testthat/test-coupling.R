test_that('nn_affine_coupling_block forward pass transforms input correctly', {
  input_size <- 4
  conditioning_size <- 2
  coupling_block <- nn_affine_coupling_block(input_size, conditioning_size)
  
  input <- torch_randn(10, input_size)
  conditioning <- torch_randn(10, conditioning_size)
  output <- coupling_block(input, conditioning)
  
  expect_equal(output$size(), input$size())
  expect_true(!is.null(attr(output, 'log_jacobian')))
})

test_that('nn_affine_coupling_block reverse pass restores input', {
  input_size <- 4
  conditioning_size <- 2
  coupling_block <- nn_affine_coupling_block(input_size, conditioning_size)
  
  input <- torch_randn(10, input_size)
  conditioning <- torch_randn(10, conditioning_size)
  output <- coupling_block(input, conditioning)
  restored_input <- coupling_block$reverse(output, conditioning)
  
  expect_equal(as_array(restored_input), as_array(input), tolerance = 1e-5)
})

test_that('nn_affine_coupling_block handles batch dimensions', {
  input_size <- 4
  conditioning_size <- 2
  n_batch <- 5
  coupling_block <- nn_affine_coupling_block(input_size, conditioning_size)
  
  input <- torch_randn(n_batch, input_size)
  conditioning <- torch_randn(n_batch, conditioning_size)
  output <- coupling_block(input, conditioning)
  
  expect_equal(output$size(), c(n_batch, input_size))
  expect_equal(attr(output, 'log_jacobian')$size(), c(n_batch, 1))
})

test_that('nn_affine_coupling_block dimension method returns correct size', {
  input_size <- 4
  coupling_block <- nn_affine_coupling_block(input_size)
  
  expect_equal(coupling_block$dimension(), input_size)
})

test_that('nn_affine_coupling_block uses custom f_scale, f_shift, g_scale, g_shift', {
  input_size <- 4
  conditioning_size <- 2
  left_size <- 2
  
  f_scale <- nn_conditional_mlp(input_size - left_size, conditioning_size, left_size)
  f_shift <- nn_conditional_mlp(input_size - left_size, conditioning_size, left_size)
  g_scale <- nn_conditional_mlp(left_size, conditioning_size, input_size - left_size)
  g_shift <- nn_conditional_mlp(left_size, conditioning_size, input_size - left_size)
  
  coupling_block <- nn_affine_coupling_block(
    input_size = input_size,
    conditioning_size = conditioning_size,
    left_size = left_size,
    f_scale = f_scale,
    f_shift = f_shift,
    g_scale = g_scale,
    g_shift = g_shift
  )
  
  input <- torch_randn(10, input_size)
  conditioning <- torch_randn(10, conditioning_size)
  output <- coupling_block(input, conditioning)
  
  expect_equal(output$size(), input$size())
  expect_true(!is.null(attr(output, 'log_jacobian')))
})