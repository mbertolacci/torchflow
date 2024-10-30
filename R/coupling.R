.torch_head <- function(x, n) {
  torch_index_select(
    x,
    -1,
    torch_tensor(1L : n, device = x$device)
  )
}

.torch_tail <- function(x, n) {
  torch_index_select(
    x,
    -1,
    torch_tensor((x$size(-1) - n + 1L) : x$size(-1), device = x$device)
  )
}

.torch_soft_clamp <- function(x, soft_clamp) {
  (2 * soft_clamp / pi) * torch_atan(x / soft_clamp)
}

#' Affine Coupling Block
#'
#' An affine coupling block is a conditional flow inheriting from
#' [nn_conditional_flow()] that applies the following transformation to the
#' input.
#'
#' Let \eqn{x = (x_1, x_2)} be a split of the input into two parts, and let
#' \eqn{u} be the conditioning input. The forward transformation is given by:
#'
#' \deqn{
#'   y_1 = x_1 \exp(f_\text{scale}(x_2, u)) + f_\text{shift}(x_2, u)
#'   y_2 = x_2 \exp(g_\text{scale}(y_1, u)) + g_\text{shift}(y_1, u)
#' }
#'
#' The inverse transformation is given by:
#'
#' \deqn{
#'   x_1 = y_1 \exp(g_\text{scale}(y_2, u)) + g_\text{shift}(y_2, u)
#'   x_2 = y_2 \exp(f_\text{scale}(x_1, u)) + f_\text{shift}(x_1, u)
#' }
#'
#' The log determinant of the Jacobian of the transformation is given by:
#'
#' \deqn{
#'   \log | \det \frac{\partial y}{\partial x} |
#'    = \sum_{i=1}^2 f_\text{scale}(x_i, u) + g_\text{scale}(y_i, u)
#' }
#'
#' By performing multiple such transformations in sequence, we can construct a
#' complex normalizing flow capable of modeling complicated conditional
#' distributions. Between each pair of such transformations, the dimensions of
#' the input should be permuted using a [nn_permutation_flow()].
#'
#' @param input_size The dimension of the input. The input itself is a tensor
#' with dimensions `[batch_size, input_size]`, or just `[input_size]` if there
#' is no batch dimension.
#' @param conditioning_size The dimension of the conditioning input, which has
#' the same batch dimensions as the input.
#' @param left_size The dimension of the left part of the input (the split
#'   \eqn{x_1} in the equations above).
#' @param f_scale The function \eqn{f_\text{scale}} in the equations above. This,
#'   and the following parameters, default to a conditional multi-layer
#'   perceptron (MLP); see [nn_conditional_mlp()]. It must inherit from
#'   [nn_conditional()].
#' @param f_shift The function \eqn{f_\text{shift}} in the equations above; see
#'   the above.
#' @param g_scale The function \eqn{g_\text{scale}} in the equations above; see
#'   the above.
#' @param g_shift The function \eqn{g_\text{shift}} in the equations above; see
#'   the above.
#' @param soft_clamp The soft clamp value for the scale parameters.
#'
#' @examples
#' library(torch)
#' # Coupling block used on its own with no conditioning
#' flow_model <- nn_affine_coupling_block(2, 0)
#' x <- torch_randn(10, 2)
#' y <- flow_model(x)
#' # y will be a tensor of dimensions [10, 2]
#' x_recovered <- flow_model$reverse(y)
#' # x_recovered will be a tensor of dimensions [10, 2]
#' # and numerically close to the original x
#' 
#' # Coupling block used with conditioning
#' flow_model <- nn_affine_coupling_block(2, 4)
#' x <- torch_randn(10, 2)
#' u <- torch_randn(10, 4)
#' y <- flow_model(x, u)
#' # y will be a tensor of dimensions [10, 2]
#' x_recovered <- flow_model$reverse(y, u)
#' # x_recovered will be a tensor of dimensions [10, 2]
#' # and numerically close to the original x
#' 
#' # Coupling block used as part of a more complex flow model
#' flow_model <- nn_sequential_conditional_flow(
#'   nn_affine_coupling_block(2, 4),
#'   nn_permutation_flow(2),
#'   nn_affine_coupling_block(2, 4)
#' )
#' y <- flow_model(x, u)
#'
#' @export
nn_affine_coupling_block <- nn_module(
  inherit = nn_conditional_flow,
  initialize = function(
    input_size,
    conditioning_size = 0,
    left_size = as.integer(input_size %/% 2),
    f_scale,
    f_shift,
    g_scale,
    g_shift,
    soft_clamp = 1.9
  ) {
    self$input_size <- input_size
    self$left_size <- left_size
    self$soft_clamp <- soft_clamp
    self$f_scale <- if (missing(f_scale)) {
      nn_conditional_mlp(input_size - left_size, conditioning_size, left_size)
    } else {
      f_scale
    }
    self$f_shift <- if (missing(f_shift)) {
      nn_conditional_mlp(input_size - left_size, conditioning_size, left_size)
    } else {
      f_shift
    }
    self$g_scale <- if (missing(g_scale)) {
      nn_conditional_mlp(left_size, conditioning_size, input_size - left_size)
    } else {
      g_scale
    }
    self$g_shift <- if (missing(g_shift)) {
      nn_conditional_mlp(left_size, conditioning_size, input_size - left_size)
    } else {
      g_shift
    }
  },
  forward = function(input, conditioning) {
    input1 <- .torch_head(input, self$left_size)
    input2 <- .torch_tail(input, self$input_size - self$left_size)
    
    scale_f <- self$f_scale(input2, conditioning)
    scale_f <- .torch_soft_clamp(scale_f, self$soft_clamp)
    shift_f <- self$f_shift(input2, conditioning)
    output1 <- input1 * torch_exp(scale_f) + shift_f
    
    scale_g <- self$g_scale(output1, conditioning)
    scale_g <- .torch_soft_clamp(scale_g, self$soft_clamp)
    shift_g <- self$g_shift(output1, conditioning)
    output2 <- input2 * torch_exp(scale_g) + shift_g
    
    output <- torch_cat(list(output1, output2), -1)    
    attr(output, 'log_jacobian') <- (
      torch_sum(scale_f, -1, keepdim = TRUE)
      + torch_sum(scale_g, -1, keepdim = TRUE)
    )
    output
  },
  reverse = function(input, conditioning) {
    input1 <- .torch_head(input, self$left_size)
    input2 <- .torch_tail(input, self$input_size - self$left_size)

    shift_g <- self$g_shift(input1, conditioning)
    scale_g <- self$g_scale(input1, conditioning)
    scale_g <- .torch_soft_clamp(scale_g, self$soft_clamp)
    output2 <- (input2 - shift_g) / torch_exp(scale_g)

    shift_f <- self$f_shift(output2, conditioning)
    scale_f <- self$f_scale(output2, conditioning)
    scale_f <- .torch_soft_clamp(scale_f, self$soft_clamp)
    output1 <- (input1 - shift_f) / torch_exp(scale_f)

    torch_cat(list(output1, output2), -1)
  },
  dimension = function() {
    self$input_size
  }
)
