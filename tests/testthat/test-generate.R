simple_model_no_conditioning <- nn_sequential_conditional_flow(
  nn_affine_coupling_block(input_size = 2),
  nn_permutation_flow(input_size = 2),
  nn_affine_coupling_block(input_size = 2)
)

simple_model_with_conditioning <- nn_sequential_conditional_flow(
  nn_affine_coupling_block(input_size = 2, conditioning_size = 3),
  nn_permutation_flow(input_size = 2),
  nn_affine_coupling_block(input_size = 2, conditioning_size = 3)
)

test_that('generate_from_conditional_flow generates correct number of samples without conditioning', {
  n_samples <- 100
  samples <- generate_from_conditional_flow(
    model = simple_model_no_conditioning,
    n_samples_per_batch = n_samples
  )
  
  expect_equal(samples$size(), c(n_samples, simple_model_no_conditioning$dimension()))
})

test_that('generate_from_conditional_flow handles conditioning', {
  n_samples <- 100
  conditioning <- torch_randn(3)
  samples <- generate_from_conditional_flow(
    model = simple_model_with_conditioning,
    n_samples_per_batch = n_samples,
    conditioning = conditioning
  )
  
  expected_size <- c(n_samples, simple_model_with_conditioning$dimension())
  expect_equal(samples$size(), expected_size)
})

test_that('generate_from_conditional_flow generates correct number of samples with batch conditioning', {
  n_samples <- 100
  conditioning <- torch_randn(10, 3)
  samples <- generate_from_conditional_flow(
    model = simple_model_with_conditioning,
    n_samples_per_batch = n_samples,
    conditioning = conditioning
  )
  
  expected_size <- c(n_samples, 10, simple_model_with_conditioning$dimension())
  expect_equal(samples$size(), expected_size)
})
