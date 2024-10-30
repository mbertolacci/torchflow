
#' Sequential Conditional Flow
#'
#' A sequential conditional flow is a conditional flow inheriting from
#' [nn_conditional_flow()] that applies a sequence of conditional flows, passing
#' the conditioning input to each flow. It is the analog of
#' [torch::nn_sequential()] for conditional flows.
#'
#' @param ... A sequence of conditional flows, or a list of conditional flows.
#'
#' @examples
#' flow_model <- nn_sequential_conditional_flow(
#'   nn_affine_coupling_block(10, 5),
#'   nn_permutation_flow(10),
#'   nn_affine_coupling_block(10, 5)
#' )
#'
#' @seealso [nn_conditional_flow()]
#' @export
nn_sequential_conditional_flow <- nn_module(
  inherit = nn_conditional_flow,
  initialize = function(...) {
    modules <- rlang::list2(...)
    for (i in seq_along(modules)) {
      self$add_module(name = i - 1, module = modules[[i]])
    }
  },
  forward = function(input, conditioning) {
    jacobian_size <- input$size()
    jacobian_size[length(jacobian_size)] <- 1
    log_jacobian <- torch_zeros(jacobian_size, device = input$device)
    output <- input
    for (module in private$modules_) {
      output <- module$forward(output, conditioning)
      log_jacobian <- log_jacobian + attr(output, 'log_jacobian')
    }
    attr(output, 'log_jacobian') <- log_jacobian
    output
  },
  reverse = function(input, conditioning) {
    output <- input
    for (module in rev(private$modules_)) {
      output <- module$reverse(output, conditioning)
    }
    output
  },
  dimension = function() {
    private$modules_[[1]]$dimension()
  }
)
