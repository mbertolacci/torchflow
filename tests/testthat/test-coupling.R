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

test_that('nn_affine_coupling_transform reports parameter count', {
  transform <- nn_affine_coupling_transform()

  expect_equal(transform$params_per_dim(), 2L)
})

test_that('nn_affine_coupling_transform maps zero parameters to identity', {
  transform <- nn_affine_coupling_transform()
  input <- torch_randn(10, 4)
  parameters <- torch_zeros(10, 8)

  output <- transform(input, parameters)
  restored_input <- transform$reverse(output, parameters)

  expect_equal(as_array(output), as_array(input), tolerance = 1e-6)
  expect_equal(as_array(restored_input), as_array(input), tolerance = 1e-6)
  expect_equal(
    as_array(attr(output, 'log_jacobian')),
    array(0, dim = c(10, 1)),
    tolerance = 1e-6
  )
})

test_that('nn_single_coupling_block transforms and reverses input', {
  input_size <- 4
  conditioning_size <- 2
  coupling_block <- nn_single_coupling_block(input_size, conditioning_size)

  input <- torch_randn(10, input_size)
  conditioning <- torch_randn(10, conditioning_size)
  output <- coupling_block(input, conditioning)
  restored_input <- coupling_block$reverse(output, conditioning)

  expect_equal(output$size(), input$size())
  expect_equal(attr(output, 'log_jacobian')$size(), c(10, 1))
  expect_equal(as_array(restored_input), as_array(input), tolerance = 1e-5)
})

test_that('nn_single_coupling_block defaults to identity for affine transform', {
  input_size <- 4
  conditioning_size <- 2
  coupling_block <- nn_single_coupling_block(input_size, conditioning_size)

  input <- torch_randn(10, input_size)
  conditioning <- torch_randn(10, conditioning_size)
  output <- coupling_block(input, conditioning)

  expect_equal(as_array(output), as_array(input), tolerance = 1e-6)
  expect_equal(
    as_array(attr(output, 'log_jacobian')),
    array(0, dim = c(10, 1)),
    tolerance = 1e-6
  )
})

test_that('nn_single_coupling_block can transform the right part', {
  input_size <- 4
  conditioning_size <- 2
  left_size <- 2
  coupling_block <- nn_single_coupling_block(
    input_size,
    conditioning_size,
    left_size,
    transform_left = FALSE
  )

  input <- torch_randn(10, input_size)
  conditioning <- torch_randn(10, conditioning_size)
  output <- coupling_block(input, conditioning)

  index <- torch_tensor(1L:left_size, device = output$device)
  expect_equal(
    as_array(torch_index_select(output, -1, index)),
    as_array(torch_index_select(input, -1, index))
  )
  expect_equal(output$size(), input$size())
  expect_true(!is.null(attr(output, 'log_jacobian')))
})

test_that('nn_single_coupling_block handles univariate input without conditioning', {
  coupling_block <- nn_single_coupling_block(
    input_size = 1
  )

  input <- torch_randn(10, 1)
  output <- coupling_block(input)
  restored_input <- coupling_block$reverse(output)

  expect_equal(output$size(), input$size())
  expect_equal(attr(output, 'log_jacobian')$size(), c(10, 1))
  expect_equal(as_array(output), as_array(input), tolerance = 1e-6)
  expect_equal(as_array(restored_input), as_array(input), tolerance = 1e-6)
})

test_that('nn_single_coupling_block handles univariate input with conditioning', {
  coupling_block <- nn_single_coupling_block(
    input_size = 1,
    conditioning_size = 2
  )

  input <- torch_randn(10, 1)
  conditioning <- torch_randn(10, 2)
  output <- coupling_block(input, conditioning)
  restored_input <- coupling_block$reverse(output, conditioning)

  expect_equal(output$size(), input$size())
  expect_equal(attr(output, 'log_jacobian')$size(), c(10, 1))
  expect_equal(as_array(output), as_array(input), tolerance = 1e-6)
  expect_equal(as_array(restored_input), as_array(input), tolerance = 1e-6)
})

test_that('nn_dual_coupling_block transforms and reverses input', {
  input_size <- 4
  conditioning_size <- 2
  coupling_block <- nn_dual_coupling_block(input_size, conditioning_size)

  input <- torch_randn(10, input_size)
  conditioning <- torch_randn(10, conditioning_size)
  output <- coupling_block(input, conditioning)
  restored_input <- coupling_block$reverse(output, conditioning)

  expect_equal(output$size(), input$size())
  expect_equal(attr(output, 'log_jacobian')$size(), c(10, 1))
  expect_equal(as_array(restored_input), as_array(input), tolerance = 1e-5)
})

test_that('nn_dual_coupling_block defaults to identity for affine transform', {
  input_size <- 4
  conditioning_size <- 2
  coupling_block <- nn_dual_coupling_block(input_size, conditioning_size)

  input <- torch_randn(10, input_size)
  conditioning <- torch_randn(10, conditioning_size)
  output <- coupling_block(input, conditioning)

  expect_equal(as_array(output), as_array(input), tolerance = 1e-6)
  expect_equal(
    as_array(attr(output, 'log_jacobian')),
    array(0, dim = c(10, 1)),
    tolerance = 1e-6
  )
})

test_that('nn_dual_coupling_block errors for univariate input', {
  expect_error(
    nn_dual_coupling_block(1),
    "`nn_dual_coupling_block()` requires `input_size` greater than 1.",
    fixed = TRUE
  )
})

test_that('nn_dual_coupling_block uses custom f_params and g_params', {
  input_size <- 4
  conditioning_size <- 2
  left_size <- 2

  f_params <- nn_conditional_mlp(
    input_size - left_size,
    conditioning_size,
    2 * left_size
  )
  g_params <- nn_conditional_mlp(
    left_size,
    conditioning_size,
    2 * (input_size - left_size)
  )

  coupling_block <- nn_dual_coupling_block(
    input_size = input_size,
    conditioning_size = conditioning_size,
    left_size = left_size,
    f_params = f_params,
    g_params = g_params
  )

  input <- torch_randn(10, input_size)
  conditioning <- torch_randn(10, conditioning_size)
  output <- coupling_block(input, conditioning)
  restored_input <- coupling_block$reverse(output, conditioning)

  expect_equal(output$size(), input$size())
  expect_equal(as_array(restored_input), as_array(input), tolerance = 1e-5)
})

test_that('nn_affine_coupling_block accepts custom f_params and g_params', {
  input_size <- 4
  conditioning_size <- 2
  left_size <- 2

  f_params <- nn_conditional_mlp(
    input_size - left_size,
    conditioning_size,
    2 * left_size
  )
  g_params <- nn_conditional_mlp(
    left_size,
    conditioning_size,
    2 * (input_size - left_size)
  )

  coupling_block <- nn_affine_coupling_block(
    input_size = input_size,
    conditioning_size = conditioning_size,
    left_size = left_size,
    f_params = f_params,
    g_params = g_params
  )

  input <- torch_randn(10, input_size)
  conditioning <- torch_randn(10, conditioning_size)
  output <- coupling_block(input, conditioning)
  restored_input <- coupling_block$reverse(output, conditioning)

  expect_equal(output$size(), input$size())
  expect_equal(as_array(restored_input), as_array(input), tolerance = 1e-5)
})

test_that('nn_affine_coupling_block warns and works for univariate input', {
  expect_warning(
    nn_affine_coupling_block(1),
    "composing affine blocks is no better than a single affine block"
  )
  coupling_block <- suppressWarnings(nn_affine_coupling_block(1))

  input <- torch_randn(10, 1)
  output <- coupling_block(input)
  restored_input <- coupling_block$reverse(output)

  expect_equal(output$size(), input$size())
  expect_equal(attr(output, 'log_jacobian')$size(), c(10, 1))
  expect_equal(as_array(output), as_array(input), tolerance = 1e-6)
  expect_equal(as_array(restored_input), as_array(input), tolerance = 1e-6)
})

test_that('nn_affine_coupling_block has trainable raw parameters for univariate input without conditioning', {
  coupling_block <- suppressWarnings(nn_affine_coupling_block(1))

  expect_true(length(coupling_block$parameters) > 0)

  input <- torch_randn(10, 1)
  output <- coupling_block(input)

  expect_equal(as_array(output), as_array(input), tolerance = 1e-6)
  expect_equal(
    as_array(attr(output, 'log_jacobian')),
    array(0, dim = c(10, 1)),
    tolerance = 1e-6
  )
})

test_that('nn_affine_coupling_block does not warn for multivariate input', {
  expect_warning(
    nn_affine_coupling_block(2),
    NA
  )
})

test_that('unknown coupling transform errors clearly', {
  expect_error(
    nn_single_coupling_block(4, transform = 'unknown'),
    "Unknown coupling transform 'unknown'.",
    fixed = TRUE
  )
})

test_that('nn_affine_coupling_block does not allow transform override', {
  expect_error(
    nn_affine_coupling_block(4, transform = 'unknown'),
    '`nn_affine_coupling_block()` always uses `transform = "affine"`.',
    fixed = TRUE
  )
})
