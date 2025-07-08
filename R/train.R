#' Train a conditional flow model
#'
#' Method to train a conditional flow model. This is a basic training loop with
#' the following steps:
#'
#' The training algorithm is as follows. For each epoch:
#' 1. Generate (using `generate`) a batch of target and conditioning samples.
#' 2. Loop over the batches of the epoch, performing a gradient descent step
#'    for each batch. The batches are processed in order from the generated
#'    samples.
#' 3. Call `after_epoch` (if provided) with the current epoch and the generated
#'    samples. This can be used to print test loss or any other tasks.
#'
#' The `generate` function (called with the current epoch number as an argument)
#' should return a list with the following elements:
#' * `target`: An [array()], [matrix()], or [torch::torch_tensor()] of target
#'    samples.
#' * `conditioning`: An optional [array()], [matrix()], or [torch::torch_tensor()]
#'    of conditioning samples.
#' If returning `torch_tensor()` objects, take care that they are on the same
#' device as the model.
#'
#' The generated samples are the choice of the user. You could generate new
#' samples each epoch, or share the same samples across epochs (noting that
#' the model may overfit in this case). In that latter case, it would be good
#' to permute the order of the samples each epoch.
#'
#' The training may be stopped early. The original model object is modified in
#' place.
#'
#' @param model A conditional flow model inheriting from [nn_conditional_flow()].
#' @param generate A function that generates a batch of target and conditioning
#'   samples; see above for details. This will be passed the current epoch number
#'   as an argument.
#' @param optimizer The optimizer to use, e.g. [torch::optim_adam()].
#' @param n_epochs The number of epochs to train for.
#' @param batch_size The batch size.
#' @param after_epoch A function to call after each epoch.
#' @param verbose Whether to print progress.
#' @param ... Additional arguments to pass to the optimizer.
#'
#' @examples
#' library(torch)
#' model <- nn_sequential_conditional_flow(
#'   nn_affine_coupling_block(input_size = 2),
#'   nn_permutation_flow(input_size = 2),
#'   nn_affine_coupling_block(input_size = 2)
#' )
#' generate <- function(epoch) {
#'   list(target = 2 + torch_randn(1024, 2))
#' }
#' # In practice, the number of epochs should be larger
#' train_conditional_flow(model, generate, n_epochs = 2)
#' @export
train_conditional_flow <- function(
  model,
  generate,
  optimizer = torch::optim_adam,
  n_epochs = 128,
  batch_size = 32,
  after_epoch = NULL,
  verbose = TRUE,
  ...
) {
  n_epochs <- as.integer(n_epochs)
  batch_size <- as.integer(batch_size)
  model_device <- model$parameters[[1]]$device
  optimizer <- match.fun(optimizer)
  optimiser <- optimizer(model$parameters, ...)
  for (epoch in seq_len(n_epochs)) {
    if (verbose) {
      cat('= Starting epoch', epoch, '\n')
    }
    samples_epoch <- generate(epoch)
    if (!inherits(samples_epoch$target, 'torch_tensor')) {
      samples_epoch$target <- torch_tensor(samples_epoch$target, device = model_device)
    }
    n_size <- samples_epoch$target$size(1)
    if (!is.null(samples_epoch$conditioning) && !inherits(samples_epoch$conditioning, 'torch_tensor')) {
      samples_epoch$conditioning <- torch_tensor(samples_epoch$conditioning, device = model_device)
      stopifnot(samples_epoch$conditioning$size(1) == n_size)
    }
    n_batches <- as.integer(ceiling(n_size / batch_size))
    if (verbose) {
      pb <- progress::progress_bar$new(total = n_batches)
    }
    for (batch in seq_len(n_batches)) {
      if (verbose) {
        pb$tick()
      }
      batch_indices <- ((batch - 1L) * batch_size + 1L) : (batch * batch_size)
      batch_indices <- torch_tensor(batch_indices[batch_indices <= n_size], device = model_device)
      batch_target <- torch_index_select(samples_epoch$target, 1, batch_indices)
      if (!is.null(samples_epoch$conditioning)) {
        batch_conditioning <- torch_index_select(samples_epoch$conditioning, 1, batch_indices)
      }
      optimiser$zero_grad()
      if (!is.null(samples_epoch$conditioning)) {
        output <- model(batch_target, batch_conditioning)
      } else {
        output <- model(batch_target)
      }
      loss <- forward_kl_loss(output)
      loss$backward()
      optimiser$step()
    }

    if (!is.null(after_epoch)) {
      after_epoch(epoch, samples_epoch)
    }
  }

  invisible(NULL)
}
