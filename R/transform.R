.torch_shifted_softplus <- function(x) {
  nnf_softplus(x + log(exp(1) - 1))
}

.resolve_coupling_transform <- function(transform, ...) {
  if (is.character(transform)) {
    if (length(transform) != 1L) {
      stop("`transform` must be a single string.", call. = FALSE)
    }
    switch(
      transform,
      affine = nn_affine_coupling_transform(...),
      stop(sprintf("Unknown coupling transform '%s'.", transform), call. = FALSE)
    )
  } else if (inherits(transform, "nn_module_generator")) {
    transform(...)
  } else if (inherits(transform, "nn_module")) {
    transform
  } else if (is.function(transform)) {
    transform(...)
  } else {
    stop(
      "`transform` must be a string, transform constructor, or transform module.",
      call. = FALSE
    )
  }
}

#' Coupling Transform
#'
#' A coupling transform maps an input partition and a set of parameters to an
#' invertible output partition. Coupling blocks use `params_per_dim()` to size
#' their parameter networks.
#'
#' @section Transform methods:
#' Subclasses should implement `params_per_dim()`, `split_parameters()`,
#' `constrain_parameters()`, `forward()`, and `reverse()`.
#'
#' @export
nn_coupling_transform <- nn_module(
  params_per_dim = function() {
    stop("Not implemented")
  },
  split_parameters = function(parameters) {
    stop("Not implemented")
  },
  constrain_parameters = function(parameters) {
    stop("Not implemented")
  },
  forward = function(input, parameters) {
    stop("Not implemented")
  },
  reverse = function(input, parameters) {
    stop("Not implemented")
  }
)

#' Affine Coupling Transform
#'
#' An affine coupling transform applies `output = input * scale + shift`.
#' Its parameter tensor contains the unconstrained scale and shift values
#' concatenated along the last dimension. The scale is constrained to be
#' positive with a shifted softplus, so a raw scale of zero maps to a
#' multiplicative scale of one.
#'
#' @param clamp Whether to apply `asinh()` to the raw scale before the shifted
#'   softplus constraint.
#'
#' @examples
#' library(torch)
#' transform <- nn_affine_coupling_transform()
#' transform$params_per_dim()
#'
#' @export
nn_affine_coupling_transform <- nn_module(
  inherit = nn_coupling_transform,
  initialize = function(clamp = TRUE) {
    self$clamp <- clamp
  },
  params_per_dim = function() {
    2L
  },
  split_parameters = function(parameters) {
    output_size <- as.integer(parameters$size(-1) / 2L)
    list(
      scale = .torch_head(parameters, output_size),
      shift = .torch_tail(parameters, output_size)
    )
  },
  constrain_parameters = function(parameters) {
    scale <- parameters$scale
    if (self$clamp) {
      scale <- torch_asinh(scale)
    }
    parameters$scale <- .torch_shifted_softplus(scale)
    parameters
  },
  forward = function(input, parameters) {
    parameters <- self$constrain_parameters(self$split_parameters(parameters))
    output <- input * parameters$scale + parameters$shift
    attr(output, "log_jacobian") <- torch_sum(
      torch_log(parameters$scale),
      -1,
      keepdim = TRUE
    )
    output
  },
  reverse = function(input, parameters) {
    parameters <- self$constrain_parameters(self$split_parameters(parameters))
    (input - parameters$shift) / parameters$scale
  }
)
