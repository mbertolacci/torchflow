#' Conditional Normalizing Flow
#'
#' A conditional normalizing flow is a normalizing flow that takes an additional
#' conditioning input. This module provides a base class for conditional
#' normalizing flows.
#'
#' The base class, `nn_conditional_flow`, is an abstract class that provides a
#' forward and reverse method, as well as a dimension method. Subclasses created
#' with [torch::nn_module()] should implement these methods. The class is a subclass
#' of [torch::nn_module()], and it inherits all of its methods and semantics.
#'
#' The base class also provides a `loss()` method compatible with [luz::setup()].
#' For luz datasets, the first batch element (`input`) is treated as the
#' conditioning input and the second (`target`) as the sample to transform. The
#' method evaluates the flow as `ctx$model(target, input)` and computes the
#' forward KL loss with [forward_kl_loss()].
#'
#' @section Forward method:
#' The forward method should return the output and the log determinant of the
#' Jacobian in the attribute `log_jacobian`. Example:
#'
#' ```r
#' forward = function(input, conditioning) {
#'   output <- ...
#'   attr(output, 'log_jacobian') <- ...
#'   output
#' }
#' ```
#'
#' @section Reverse method:
#'
#' The reverse method should return the inverse of the output, but need not
#' implement a log determinant. Example:
#'
#' ```r
#' reverse = function(input, conditioning) {
#'   output <- ...
#'   output
#' }
#' ```
#'
#' @section Dimension method:
#'
#' The dimension method should return the dimension of the input and output of
#' the flow. Example:
#'
#' ```r
#' dimension = function() {
#'   return(2)
#' }
#' ```
#'
#' @seealso [nn_summarizing_conditional_flow()], [nn_sequential_conditional_flow()],
#' [nn_permutation_flow()], and [nn_affine_coupling_block()] for subclasses.
#'
#' @export
nn_conditional_flow <- nn_module(
  forward = function(input, conditioning) {
    stop('Not implemented')
  },
  loss = function(input, target) {
    output <- ctx$model(target, input)
    forward_kl_loss(output)
  },
  reverse = function(input, conditioning) {
    stop('Not implemented')
  },
  dimension = function() {
    stop('Not implemented')
  }
)
