test_that('nn_actnorm_block initializes to identity', {
  actnorm_block <- nn_actnorm_block(3)
  input <- torch_randn(10, 3)
  output <- actnorm_block(input)
  restored_input <- actnorm_block$reverse(output)

  expect_equal(output$size(), input$size())
  expect_equal(as_array(output), as_array(input), tolerance = 1e-6)
  expect_equal(as_array(restored_input), as_array(input), tolerance = 1e-6)
  expect_equal(
    as_array(attr(output, 'log_jacobian')),
    array(0, dim = c(10, 1)),
    tolerance = 1e-6
  )
})

test_that('nn_actnorm_block reverses learned scale and bias', {
  actnorm_block <- nn_actnorm_block(3)
  with_no_grad({
    invisible(actnorm_block$scale$copy_(torch_tensor(c(2, 3, 4))))
    invisible(actnorm_block$bias$copy_(torch_tensor(c(1, -1, 0.5))))
  })

  input <- torch_randn(10, 3)
  output <- actnorm_block(input)
  restored_input <- actnorm_block$reverse(output)
  expected_log_jacobian <- log(24)

  expect_equal(as_array(restored_input), as_array(input), tolerance = 1e-6)
  expect_equal(
    as_array(attr(output, 'log_jacobian')),
    array(expected_log_jacobian, dim = c(10, 1)),
    tolerance = 1e-6
  )
})

test_that('nn_actnorm_block handles unbatched and extra batch dimensions', {
  actnorm_block <- nn_actnorm_block(2)

  input <- torch_randn(2)
  output <- actnorm_block(input)
  expect_equal(output$size(), input$size())
  expect_equal(attr(output, 'log_jacobian')$size(), 1)

  input <- torch_randn(3, 4, 2)
  output <- actnorm_block(input)
  restored_input <- actnorm_block$reverse(output)

  expect_equal(output$size(), input$size())
  expect_equal(attr(output, 'log_jacobian')$size(), c(3, 4, 1))
  expect_equal(as_array(restored_input), as_array(input), tolerance = 1e-6)
})

test_that('nn_actnorm_block ignores conditioning', {
  actnorm_block <- nn_actnorm_block(2)

  input <- torch_randn(10, 2)
  conditioning <- torch_randn(10, 3)
  output <- actnorm_block(input, conditioning)
  restored_input <- actnorm_block$reverse(output, conditioning)

  expect_equal(output$size(), input$size())
  expect_equal(attr(output, 'log_jacobian')$size(), c(10, 1))
  expect_equal(as_array(restored_input), as_array(input), tolerance = 1e-6)
})

test_that('nn_actnorm_block exposes trainable parameters and dimension', {
  actnorm_block <- nn_actnorm_block(4)

  expect_equal(actnorm_block$dimension(), 4)
  expect_equal(length(actnorm_block$parameters), 2)
})
