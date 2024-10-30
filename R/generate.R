#' Generate samples from a conditional flow model
#' 
#' This function generates samples from a conditional flow model. If
#' conditioning is provided, `n_samples_per_batch` samples are generated for
#' each batch of the conditioning variable. If conditioning is not provided,
#' `n_samples_per_batch` samples are generated.
#' 
#' @param model A conditional flow model.
#' @param n_samples_per_batch The number of samples to generate for each batch
#' of the conditioning variable, or the total number of samples if conditioning
#' is not provided.
#' @param conditioning The conditioning variable, a torch tensor of dimensions
#' `[batch, ...]` where `batch` is the dimension of the batch and `...` are the
#' dimensions of the conditioning variable.
#' 
#' @export
generate_from_conditional_flow <- function(
  model,
  n_samples_per_batch,
  conditioning
) {
  drop_batch_dimension <- FALSE
  if (missing(conditioning)) {
    batch_size <- 1L
    drop_batch_dimension <- TRUE
  } else {
    conditioning_size <- conditioning$size()
    if (length(conditioning_size) == 1) {
      drop_batch_dimension <- TRUE
      batch_size <- 1
      conditioning <- conditioning$unsqueeze(1)
    } else {
      batch_size <- conditioning$size(1)
    }
  }

  n_samples <- batch_size * n_samples_per_batch
  if (!missing(conditioning)) {
    conditioning_size <- conditioning$size()
    conditioning <- conditioning$unsqueeze(
      1
    )$expand(
      c(n_samples_per_batch, conditioning_size)
    )$reshape(
      c(n_samples, conditioning_size[-1])
    )
  }
  
  z_samples <- torch_randn(c(n_samples, model$dimension()), device = model$parameters[[1]]$device)
  model$reverse(z_samples, conditioning)$reshape(
    if (drop_batch_dimension) {
      c(n_samples_per_batch, model$dimension())
    } else {
      c(n_samples_per_batch, batch_size, model$dimension())
    }
  )
}
