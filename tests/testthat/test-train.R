simple_model <- nn_sequential_conditional_flow(
  nn_affine_coupling_block(input_size = 2),
  nn_permutation_flow(input_size = 2),
  nn_affine_coupling_block(input_size = 2)
)

generate_samples <- function(epoch) {
  list(target = 2 + torch_randn(64, 2))
}

simple_model_with_conditioning <- nn_sequential_conditional_flow(
  nn_affine_coupling_block(input_size = 2, conditioning_size = 2),
  nn_permutation_flow(input_size = 2),
  nn_affine_coupling_block(input_size = 2, conditioning_size = 2)
)

generate_samples_with_conditioning <- function(epoch) {
  list(
    target = 2 + torch_randn(64, 2),
    conditioning = torch_randn(64, 2)
  )
}

test_that('train_conditional_flow runs without error', {
  expect_silent(
    train_conditional_flow(
      model = simple_model,
      generate = generate_samples,
      n_epochs = 1,
      batch_size = 16,
      verbose = FALSE
    )
  )
})

test_that('train_conditional_flow updates model parameters', {
  # Capture initial parameters
  initial_params <- lapply(simple_model$parameters, function(p) p$clone())
  
  # Train the model
  train_conditional_flow(
    model = simple_model,
    generate = generate_samples,
    n_epochs = 1,
    batch_size = 16,
    verbose = FALSE
  )
  
  # Check that at least one parameter has changed
  params_changed <- sapply(seq_along(simple_model$parameters), function(i) {
    !torch_allclose(initial_params[[i]], simple_model$parameters[[i]])
  })
  
  expect_true(any(params_changed))
})

test_that('train_conditional_flow handles after_epoch callback', {
  after_epoch_called <- FALSE
  after_epoch_function <- function(epoch, samples) {
    after_epoch_called <<- TRUE
  }
  
  train_conditional_flow(
    model = simple_model,
    generate = generate_samples,
    n_epochs = 1,
    batch_size = 16,
    after_epoch = after_epoch_function,
    verbose = FALSE
  )
  
  expect_true(after_epoch_called)
})

test_that('train_conditional_flow handles conditioning input', {
  expect_silent(
    train_conditional_flow(
      model = simple_model_with_conditioning,
      generate = generate_samples_with_conditioning,
      n_epochs = 1,
      batch_size = 16,
      verbose = FALSE
    )
  )
})

test_that('train_conditional_flow handles samples that are not tensors', {
  generate_samples_not_tensor <- function(epoch) {
    list(
      target = 2 + matrix(rnorm(64 * 2), nrow = 64, ncol = 2),
      conditioning = matrix(rnorm(64 * 2), nrow = 64, ncol = 2)
    )
  }
  expect_silent(
    train_conditional_flow(
      model = simple_model_with_conditioning,
      generate = generate_samples_not_tensor,
      n_epochs = 1,
      batch_size = 16,
      verbose = FALSE
    )
  )
})

test_that('train_conditional_flow runs with verbose output', {
  expect_output(
    train_conditional_flow(
      model = simple_model,
      generate = generate_samples,
      n_epochs = 1,
      batch_size = 16,
      verbose = TRUE
    )
  )
})

test_that('train_conditional_flow works with different optimizer', {
  expect_silent(
    train_conditional_flow(
      model = simple_model,
      generate = generate_samples,
      optimizer = torch::optim_sgd,
      n_epochs = 1,
      batch_size = 16,
      verbose = FALSE,
      lr = 0.01
    )
  )
})

test_that('train_conditional_flow handles zero epochs gracefully', {
  expect_silent(
    train_conditional_flow(
      model = simple_model,
      generate = generate_samples,
      n_epochs = 0,
      batch_size = 16,
      verbose = FALSE
    )
  )
})

test_that('train_conditional_flow handles large batch size', {
  expect_silent(
    train_conditional_flow(
      model = simple_model,
      generate = generate_samples,
      n_epochs = 1,
      batch_size = 128,  # Larger than the dataset
      verbose = FALSE
    )
  )
})
