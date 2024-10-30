library(testthat)

make_base_summarizing_flow <- function(input_size, conditioning_size, summary_size) {
  summary_model <- nn_sequential(
    nn_linear(conditioning_size, summary_size),
    nn_relu()
  )
  flow_model <- nn_affine_coupling_block(input_size, summary_size)
  nn_summarizing_conditional_flow(summary_model, flow_model)
}

test_that('nn_summarizing_conditional_flow processes input with summary model', {
  input_size <- 10
  conditioning_size <- 10
  summary_size <- 5
  
  summarizing_flow <- make_base_summarizing_flow(input_size, conditioning_size, summary_size)
  
  input <- torch_randn(input_size)
  conditioning <- torch_randn(conditioning_size)
  output <- summarizing_flow(input, conditioning)
  
  expect_equal(output$size(), input$size())
  expect_equal(attr(output, 'log_jacobian')$size(), 1)
})

test_that('nn_summarizing_conditional_flow reverse pass restores input', {
  input_size <- 10
  conditioning_size <- 10
  summary_size <- 5
  
  summarizing_flow <- make_base_summarizing_flow(input_size, conditioning_size, summary_size)
  
  input <- torch_randn(input_size)
  conditioning <- torch_randn(conditioning_size)
  output <- summarizing_flow(input, conditioning)
  restored_input <- summarizing_flow$reverse(output, conditioning)
  
  expect_equal(as_array(restored_input), as_array(input), tolerance = 1e-5)
})

test_that('nn_summarizing_conditional_flow works with batch dimensions', {
  n_batch <- 8
  input_size <- 10
  conditioning_size <- 10
  summary_size <- 5
  
  summarizing_flow <- make_base_summarizing_flow(input_size, conditioning_size, summary_size)
  
  input <- torch_randn(n_batch, input_size)
  conditioning <- torch_randn(n_batch, conditioning_size)
  output <- summarizing_flow(input, conditioning)
  
  expect_equal(output$size(), input$size())
  expect_equal(attr(output, 'log_jacobian')$size(), c(n_batch, 1))
})

test_that('nn_summarizing_conditional_flow uses precomputed summary', {
  input_size <- 10
  conditioning_size <- 10
  summary_size <- 5
  
  summarizing_flow <- make_base_summarizing_flow(input_size, conditioning_size, summary_size)
  
  input <- torch_randn(input_size)
  conditioning <- torch_randn(conditioning_size)
  precomputed_summary <- summarizing_flow$summarize(conditioning)
  
  output_with_summary <- summarizing_flow(input, conditioning, summary = precomputed_summary)
  output_without_summary <- summarizing_flow(input, conditioning)
  
  expect_equal(as_array(output_with_summary), as_array(output_without_summary))
})

test_that('nn_summarizing_conditional_flow dimension method returns correct size', {
  input_size <- 10
  conditioning_size <- 10
  summary_size <- 5
  summarizing_flow <- make_base_summarizing_flow(input_size, conditioning_size, summary_size)
  
  expect_equal(summarizing_flow$dimension(), input_size)
})
