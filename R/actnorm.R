#' ActNorm Block
#'
#' An ActNorm block is a conditional flow inheriting from
#' [nn_conditional_flow()] that applies a learned per-dimension affine
#' transformation,
#'
#' \deqn{y = s x + b}
#'
#' where `s` and `b` are trainable vectors initialized to one and zero. The
#' block ignores conditioning inputs and can be inserted directly into
#' [nn_sequential_conditional_flow()].
#'
#' @param input_size The size of the input to the flow.
#'
#' @examples
#' library(torch)
#' flow_model <- nn_sequential_conditional_flow(
#'   nn_actnorm_block(2),
#'   nn_spline_coupling_block(2),
#'   nn_permutation_flow(2),
#'   nn_actnorm_block(2),
#'   nn_spline_coupling_block(2)
#' )
#'
#' @seealso [nn_conditional_flow()]
#' @include conditional-flow.R
#' @export
nn_actnorm_block <- nn_module(
  inherit = nn_conditional_flow,
  initialize = function(input_size) {
    self$input_size <- as.integer(input_size)
    self$scale <- nn_parameter(torch_ones(self$input_size))
    self$bias <- nn_parameter(torch_zeros(self$input_size))
  },
  forward = function(input, ...) {
    jacobian_size <- input$size()
    jacobian_size[length(jacobian_size)] <- 1L

    output <- input * self$scale + self$bias
    log_jacobian <- torch_sum(torch_log(torch_abs(self$scale)))
    attr(output, "log_jacobian") <- log_jacobian$expand(jacobian_size)
    output
  },
  reverse = function(output, ...) {
    (output - self$bias) / self$scale
  },
  dimension = function() {
    self$input_size
  }
)
