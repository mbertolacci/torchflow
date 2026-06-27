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

test_that('nn_spline_coupling_block handles values outside spline tails', {
  coupling_block <- nn_spline_coupling_block(4)
  input <- torch_tensor(matrix(c(
    -4, 0, 0, 4,
    3, -3, 3.5, -3.5
  ), nrow = 2, byrow = TRUE))

  output <- coupling_block(input)
  restored_input <- coupling_block$reverse(output)

  expect_equal(output$size(), input$size())
  expect_true(as.logical(torch_allclose(input, restored_input, atol = 1e-4)))
})

test_that('nn_spline_coupling_block propagates gradients through forward transform', {
  torch_manual_seed(1)
  coupling_block <- nn_spline_coupling_block(4, 2)
  input <- torch_tensor(matrix(c(
    -0.8, -0.2, 0.3, 0.7,
    0.4, -0.6, 0.1, -0.5
  ), nrow = 2, byrow = TRUE), requires_grad = TRUE)
  conditioning <- torch_tensor(matrix(c(
    0.2, -0.3,
    0.5, 0.1
  ), nrow = 2, byrow = TRUE), requires_grad = TRUE)

  output <- coupling_block(input, conditioning)
  loss <- output$sum() + attr(output, 'log_jacobian')$sum()
  loss$backward()

  expect_false(is.null(input$grad))
  expect_false(is.null(conditioning$grad))
  expect_true(as.logical(torch_isfinite(input$grad)$all()))
  expect_true(as.logical(torch_isfinite(conditioning$grad)$all()))
  expect_gt(as.numeric(input$grad$abs()$sum()), 0)
  expect_gt(as.numeric(conditioning$grad$abs()$sum()), 0)

  parameter_grads <- lapply(coupling_block$parameters, function(parameter) parameter$grad)
  expect_false(any(vapply(parameter_grads, is.null, logical(1))))
  expect_true(all(vapply(parameter_grads, function(grad) as.logical(torch_isfinite(grad)$all()), logical(1))))
  expect_gt(sum(vapply(parameter_grads, function(grad) as.numeric(grad$abs()$sum()), numeric(1))), 0)
})

test_that('nn_spline_coupling_block propagates gradients through reverse transform', {
  torch_manual_seed(2)
  coupling_block <- nn_spline_coupling_block(4, 2)
  input <- torch_tensor(matrix(c(
    -0.4, 0.6, -0.1, 0.2,
    0.8, -0.7, 0.5, -0.3
  ), nrow = 2, byrow = TRUE), requires_grad = TRUE)
  conditioning <- torch_tensor(matrix(c(
    -0.2, 0.4,
    0.3, -0.5
  ), nrow = 2, byrow = TRUE), requires_grad = TRUE)

  output <- coupling_block$reverse(input, conditioning)
  loss <- output$sum()
  loss$backward()

  expect_false(is.null(input$grad))
  expect_false(is.null(conditioning$grad))
  expect_true(as.logical(torch_isfinite(input$grad)$all()))
  expect_true(as.logical(torch_isfinite(conditioning$grad)$all()))
  expect_gt(as.numeric(input$grad$abs()$sum()), 0)
  expect_gt(as.numeric(conditioning$grad$abs()$sum()), 0)

  parameter_grads <- lapply(coupling_block$parameters, function(parameter) parameter$grad)
  expect_false(any(vapply(parameter_grads, is.null, logical(1))))
  expect_true(all(vapply(parameter_grads, function(grad) as.logical(torch_isfinite(grad)$all()), logical(1))))
  expect_gt(sum(vapply(parameter_grads, function(grad) as.numeric(grad$abs()$sum()), numeric(1))), 0)
})
