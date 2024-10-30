#' Permutation Flow
#'
#' A permutation flow is a conditional flow inheriting from
#' [nn_conditional_flow()] that permutes the input dimensions. The permutation is
#' fixed to a random permutation at initialization and does not change. It's
#' log Jacobian is zero since it is a simple reordering of the input dimensions.
#'
#' @param input_size The size of the input to the flow.
#'
#' @examples
#' library(torch)
#' # Use on its own
#' permutation_flow <- nn_permutation_flow(10)
#' input <- torch_randn(10)
#' output <- permutation_flow(input)
#' # Use in a more complex conditional flow
#' flow_model <- nn_sequential_conditional_flow(
#'   nn_affine_coupling_block(10, 5),
#'   nn_permutation_flow(10),
#'   nn_affine_coupling_block(10, 5)
#' )
#' @seealso [nn_conditional_flow()]
#' @export
nn_permutation_flow <- nn_module(
  inherit = nn_conditional_flow,
  initialize = function(input_size) {
    self$input_size <- as.integer(input_size)
    self$permutation <- torch_randperm(input_size) + 1L
    self$reverse_permutation <- torch_argsort(self$permutation)

    self$register_buffer('permutation', self$permutation)
    self$register_buffer('reverse_permutation', self$reverse_permutation)
  },
  forward = function(input, ...) {
    jacobian_size <- input$size()
    jacobian_size[length(jacobian_size)] <- 1
    output <- torch_index_select(input, -1, self$permutation)
    attr(output, 'log_jacobian') <- torch_zeros(jacobian_size, device = input$device)
    output
  },
  reverse = function(output, ...) {
    torch_index_select(output, -1, self$reverse_permutation)
  },
  dimension = function() {
    self$input_size
  }
)
