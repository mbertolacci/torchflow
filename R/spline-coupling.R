.torch_select_last <- function(x, indices) {
  torch_index_select(x, -1, torch_tensor(indices, dtype = torch_long(), device = x$device))
}

.torch_expand_last <- function(x, n) {
  target_size <- c(x$size(), n)
  x$unsqueeze(-1)$expand(target_size)
}

.torch_gather_last <- function(x, indices) {
  torch_gather(x, -1, indices$to(dtype = torch_long()))$squeeze(-1)
}

.torch_positive_spline_params <- function(
  unnormalized_widths,
  unnormalized_heights,
  unnormalized_derivatives,
  num_bins,
  tail_bound,
  min_bin_width,
  min_bin_height,
  min_derivative
) {
  if (min_bin_width * num_bins >= 1) {
    stop('min_bin_width * num_bins must be less than 1')
  }
  if (min_bin_height * num_bins >= 1) {
    stop('min_bin_height * num_bins must be less than 1')
  }

  widths <- nnf_softmax(unnormalized_widths, dim = -1)
  widths <- min_bin_width + (1 - min_bin_width * num_bins) * widths
  widths <- 2 * tail_bound * widths
  cumwidths <- torch_cumsum(widths, dim = -1)
  cumwidths <- torch_cat(list(
    torch_full_like(.torch_select_last(cumwidths, 1), -tail_bound),
    cumwidths - tail_bound
  ), dim = -1)
  widths <- .torch_select_last(cumwidths, 2:(num_bins + 1)) - .torch_select_last(cumwidths, 1:num_bins)

  heights <- nnf_softmax(unnormalized_heights, dim = -1)
  heights <- min_bin_height + (1 - min_bin_height * num_bins) * heights
  heights <- 2 * tail_bound * heights
  cumheights <- torch_cumsum(heights, dim = -1)
  cumheights <- torch_cat(list(
    torch_full_like(.torch_select_last(cumheights, 1), -tail_bound),
    cumheights - tail_bound
  ), dim = -1)
  heights <- .torch_select_last(cumheights, 2:(num_bins + 1)) - .torch_select_last(cumheights, 1:num_bins)

  derivatives <- min_derivative + nnf_softplus(unnormalized_derivatives)

  list(
    widths = widths,
    cumwidths = cumwidths,
    heights = heights,
    cumheights = cumheights,
    derivatives = derivatives
  )
}

.torch_rational_quadratic_spline <- function(
  inputs,
  unnormalized_widths,
  unnormalized_heights,
  unnormalized_derivatives,
  inverse = FALSE,
  tail_bound = 3,
  min_bin_width = 1e-3,
  min_bin_height = 1e-3,
  min_derivative = 1e-3
) {
  num_bins <- unnormalized_widths$size(-1)
  params <- .torch_positive_spline_params(
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    num_bins,
    tail_bound,
    min_bin_width,
    min_bin_height,
    min_derivative
  )

  inside_mask <- inputs >= -tail_bound & inputs <= tail_bound
  outputs <- inputs$clone()
  logabsdet <- torch_zeros_like(inputs)

  bin_locations <- if (inverse) params$cumheights else params$cumwidths
  bin_idx <- torch_sum(.torch_expand_last(inputs, num_bins) >= .torch_select_last(bin_locations, 2:(num_bins + 1)), dim = -1, keepdim = TRUE) + 1L
  bin_idx <- torch_clamp(bin_idx, min = 1, max = num_bins)

  input_cumwidths <- .torch_gather_last(params$cumwidths, bin_idx)
  input_bin_widths <- .torch_gather_last(params$widths, bin_idx)
  input_cumheights <- .torch_gather_last(params$cumheights, bin_idx)
  input_heights <- .torch_gather_last(params$heights, bin_idx)
  input_delta <- input_heights / input_bin_widths
  input_derivatives <- .torch_gather_last(params$derivatives, bin_idx)
  input_derivatives_plus_one <- .torch_gather_last(params$derivatives, bin_idx + 1L)

  if (inverse) {
    a <- (inputs - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * input_delta) + input_heights * (input_delta - input_derivatives)
    b <- input_heights * input_derivatives - (inputs - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
    c <- -input_delta * (inputs - input_cumheights)
    discriminant <- torch_clamp(b^2 - 4 * a * c, min = 0)
    theta <- (2 * c) / (-b - torch_sqrt(discriminant))
    theta <- torch_clamp(theta, min = 0, max = 1)
    transformed <- input_cumwidths + theta * input_bin_widths
  } else {
    theta <- (inputs - input_cumwidths) / input_bin_widths
    theta <- torch_clamp(theta, min = 0, max = 1)
    numerator <- input_heights * (input_delta * theta^2 + input_derivatives * theta * (1 - theta))
    denominator <- input_delta + (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta * (1 - theta)
    transformed <- input_cumheights + numerator / denominator
  }

  derivative_numerator <- input_delta^2 * (
    input_derivatives_plus_one * theta^2 +
      2 * input_delta * theta * (1 - theta) +
      input_derivatives * (1 - theta)^2
  )
  derivative_denominator <- input_delta + (
    input_derivatives + input_derivatives_plus_one - 2 * input_delta
  ) * theta * (1 - theta)
  lad <- torch_log(derivative_numerator) - 2 * torch_log(derivative_denominator)
  if (inverse) {
    lad <- -lad
  }

  outputs <- torch_where(inside_mask, transformed, outputs)
  logabsdet <- torch_where(inside_mask, lad, logabsdet)
  list(outputs = outputs, logabsdet = logabsdet)
}

.torch_spline_param_split <- function(raw_params, transformed_size, num_bins) {
  raw_params <- raw_params$reshape(c(raw_params$size()[1:(length(raw_params$size()) - 1)], transformed_size, 3 * num_bins + 1))
  widths <- .torch_select_last(raw_params, 1:num_bins)
  heights <- .torch_select_last(raw_params, (num_bins + 1):(2 * num_bins))
  derivatives <- .torch_select_last(raw_params, (2 * num_bins + 1):(3 * num_bins + 1))
  list(widths = widths, heights = heights, derivatives = derivatives)
}

.torch_spline_transform <- function(inputs, raw_params, inverse, tail_bound, min_bin_width, min_bin_height, min_derivative) {
  transformed_size <- inputs$size(-1)
  num_bins <- as.integer((raw_params$size(-1) / transformed_size - 1) / 3)
  params <- .torch_spline_param_split(raw_params, transformed_size, num_bins)
  result <- .torch_rational_quadratic_spline(
    inputs,
    params$widths,
    params$heights,
    params$derivatives,
    inverse = inverse,
    tail_bound = tail_bound,
    min_bin_width = min_bin_width,
    min_bin_height = min_bin_height,
    min_derivative = min_derivative
  )
  result
}

#' Spline Coupling Block
#'
#' A spline coupling block is a conditional flow inheriting from
#' [nn_conditional_flow()] that applies monotonic rational-quadratic spline
#' transformations to each side of a split input. It is a more flexible
#' alternative to [nn_affine_coupling_block()]. Values outside
#' `[-tail_bound, tail_bound]` are transformed as the identity.
#'
#' @param input_size The dimension of the input.
#' @param conditioning_size The dimension of the conditioning input.
#' @param left_size The dimension of the left part of the input split.
#' @param num_bins The number of bins in each rational-quadratic spline.
#' @param f_params,g_params Conditional networks producing spline parameters.
#' @param tail_bound Boundary of the interval transformed by the spline.
#' @param min_bin_width,min_bin_height,min_derivative Lower bounds for spline parameters.
#'
#' @examples
#' library(torch)
#' flow_model <- nn_spline_coupling_block(2, 0)
#' x <- torch_randn(10, 2)
#' y <- flow_model(x)
#' x_recovered <- flow_model$reverse(y)
#'
#' @export
nn_spline_coupling_block <- nn_module(
  inherit = nn_conditional_flow,
  initialize = function(
    input_size,
    conditioning_size = 0,
    left_size = floor(input_size / 2),
    num_bins = 8,
    f_params,
    g_params,
    tail_bound = 3,
    min_bin_width = 1e-3,
    min_bin_height = 1e-3,
    min_derivative = 1e-3
  ) {
    self$input_size <- input_size
    self$left_size <- left_size
    self$num_bins <- num_bins
    self$tail_bound <- tail_bound
    self$min_bin_width <- min_bin_width
    self$min_bin_height <- min_bin_height
    self$min_derivative <- min_derivative
    per_dimension_params <- 3 * num_bins + 1
    self$f_params <- if (missing(f_params)) {
      nn_conditional_mlp(input_size - left_size, conditioning_size, left_size * per_dimension_params)
    } else {
      f_params
    }
    self$g_params <- if (missing(g_params)) {
      nn_conditional_mlp(left_size, conditioning_size, (input_size - left_size) * per_dimension_params)
    } else {
      g_params
    }
  },
  forward = function(input, conditioning) {
    input1 <- .torch_head(input, self$left_size)
    input2 <- .torch_tail(input, self$input_size - self$left_size)

    f_raw_params <- self$f_params(input2, conditioning)
    f_result <- .torch_spline_transform(input1, f_raw_params, FALSE, self$tail_bound, self$min_bin_width, self$min_bin_height, self$min_derivative)
    output1 <- f_result$outputs

    g_raw_params <- self$g_params(output1, conditioning)
    g_result <- .torch_spline_transform(input2, g_raw_params, FALSE, self$tail_bound, self$min_bin_width, self$min_bin_height, self$min_derivative)
    output2 <- g_result$outputs

    output <- torch_cat(list(output1, output2), -1)
    attr(output, 'log_jacobian') <- torch_sum(f_result$logabsdet, -1, keepdim = TRUE) + torch_sum(g_result$logabsdet, -1, keepdim = TRUE)
    output
  },
  reverse = function(input, conditioning) {
    input1 <- .torch_head(input, self$left_size)
    input2 <- .torch_tail(input, self$input_size - self$left_size)

    g_raw_params <- self$g_params(input1, conditioning)
    g_result <- .torch_spline_transform(input2, g_raw_params, TRUE, self$tail_bound, self$min_bin_width, self$min_bin_height, self$min_derivative)
    output2 <- g_result$outputs

    f_raw_params <- self$f_params(output2, conditioning)
    f_result <- .torch_spline_transform(input1, f_raw_params, TRUE, self$tail_bound, self$min_bin_width, self$min_bin_height, self$min_derivative)
    output1 <- f_result$outputs

    torch_cat(list(output1, output2), -1)
  },
  dimension = function() {
    self$input_size
  }
)
