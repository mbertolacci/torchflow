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
      spline = nn_spline_coupling_transform(...),
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

.torch_last_dim_slice <- function(x, start, length) {
  if (length == 0L) {
    output_size <- x$size()
    output_size[length(output_size)] <- 0L
    return(torch_empty(output_size, device = x$device, dtype = x$dtype))
  }

  torch_narrow(x, -1L, start, length)
}

.torch_spline_gather <- function(x, index) {
  torch_gather(x, -1L, index + 1L)
}

.torch_rational_quadratic_spline <- function(
  x,
  left,
  right,
  bottom,
  top,
  derivative_left,
  derivative_right,
  inverse = FALSE
) {
  dx <- right - left
  dy <- top - bottom
  sk <- dy / dx

  if (!inverse) {
    xi <- (x - left) / dx

    numerator <- dy * (sk * xi^2 + derivative_left * xi * (1 - xi))
    denominator <- sk +
      (derivative_right + derivative_left - 2 * sk) * xi * (1 - xi)
    result <- bottom + numerator / denominator
  } else {
    y <- x

    a <- dy * (sk - derivative_left) +
      (y - bottom) * (derivative_right + derivative_left - 2 * sk)
    b <- dy * derivative_left -
      (y - bottom) * (derivative_right + derivative_left - 2 * sk)
    c <- -sk * (y - bottom)

    discriminant <- torch_clamp(b^2 - 4 * a * c, min = 0)
    xi <- 2 * c / (-b - torch_sqrt(discriminant))
    result <- xi * dx + left
  }

  numerator <- sk^2 * (
    derivative_right * xi^2 +
      2 * sk * xi * (1 - xi) +
      derivative_left * (1 - xi)^2
  )
  denominator <- (
    sk + (derivative_right + derivative_left - 2 * sk) * xi * (1 - xi)
  )^2
  log_jacobian <- torch_log(numerator) - torch_log(denominator)

  if (inverse) {
    log_jacobian <- -log_jacobian
  }

  list(output = result, log_jacobian = log_jacobian)
}

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

#' Spline Coupling Transform
#'
#' A rational-quadratic spline coupling transform following the BayesFlow
#' parameterization. The transform learns a monotone spline on a rectangular
#' domain and applies the affine map implied by that rectangle outside the
#' domain.
#'
#' Its parameter tensor contains, for each transformed dimension, raw values for
#' the left edge, bottom edge, total width, total height, bin widths, bin
#' heights, and interior derivatives. With zero raw parameters, the default
#' domain gives an identity transform.
#'
#' @param bins The number of spline bins.
#' @param default_domain A numeric vector `c(left, right, bottom, top)` giving
#'   the default spline domain.
#' @param min_width The minimum total width of the learned domain.
#' @param min_height The minimum total height of the learned domain.
#' @param min_bin_width The minimum width of each bin.
#' @param min_bin_height The minimum height of each bin.
#' @param method The spline method. Currently only `"rational_quadratic"` is
#'   supported.
#'
#' @examples
#' library(torch)
#' transform <- nn_spline_coupling_transform(bins = 8)
#' transform$params_per_dim()
#'
#' @export
nn_spline_coupling_transform <- nn_module(
  inherit = nn_coupling_transform,
  initialize = function(
    bins = 16,
    default_domain = c(-3, 3, -3, 3),
    min_width = 1,
    min_height = 1,
    min_bin_width = 0.1,
    min_bin_height = 0.1,
    method = "rational_quadratic"
  ) {
    bins <- as.integer(bins)
    if (bins <= 0L) {
      stop("`bins` must be strictly positive.", call. = FALSE)
    }
    if (length(default_domain) != 4L) {
      stop(
        "`default_domain` must be `c(left, right, bottom, top)`.",
        call. = FALSE
      )
    }
    if (default_domain[2] <= default_domain[1] ||
        default_domain[4] <= default_domain[3]) {
      stop(
        "`default_domain` must satisfy left < right and bottom < top.",
        call. = FALSE
      )
    }
    if (method != "rational_quadratic") {
      stop(
        'Currently, only `method = "rational_quadratic"` is supported.',
        call. = FALSE
      )
    }

    self$bins <- bins
    self$min_width <- max(min_width, bins * min_bin_width)
    self$min_height <- max(min_height, bins * min_bin_height)
    self$min_bin_width <- min_bin_width
    self$min_bin_height <- min_bin_height
    self$method <- method

    self$default_left <- default_domain[1]
    self$default_bottom <- default_domain[3]
    self$default_width <- default_domain[2] - default_domain[1]
    self$default_height <- default_domain[4] - default_domain[3]

    if (self$default_width < self$min_width) {
      stop(
        sprintf(
          "`default_domain` width must be greater than or equal to %.6g.",
          self$min_width
        ),
        call. = FALSE
      )
    }
    if (self$default_height < self$min_height) {
      stop(
        sprintf(
          "`default_domain` height must be greater than or equal to %.6g.",
          self$min_height
        ),
        call. = FALSE
      )
    }

    self$parameter_sizes <- list(
      left_edge = 1L,
      bottom_edge = 1L,
      total_width = 1L,
      total_height = 1L,
      bin_widths = self$bins,
      bin_heights = self$bins,
      derivatives = self$bins - 1L
    )
    self$shift <- sinh(1) * log(exp(1) - 1)
  },
  params_per_dim = function() {
    3L * self$bins + 3L
  },
  split_parameters = function(parameters) {
    parameter_size <- self$params_per_dim()
    output_size <- as.integer(parameters$size(-1) / parameter_size)
    batch_size <- parameters$size()
    batch_size <- batch_size[-length(batch_size)]
    parameters <- parameters$reshape(c(batch_size, output_size, parameter_size))

    start <- 1L
    split <- list()
    for (name in names(self$parameter_sizes)) {
      size <- self$parameter_sizes[[name]]
      split[[name]] <- .torch_last_dim_slice(parameters, start, size)
      start <- start + size
    }

    split
  },
  constrain_parameters = function(parameters) {
    left_edge <- parameters$left_edge + self$default_left
    bottom_edge <- parameters$bottom_edge + self$default_bottom

    total_width <- torch_asinh(nnf_softplus(parameters$total_width + self$shift))
    total_width <- (self$default_width - self$min_width) * total_width +
      self$min_width
    total_height <- torch_asinh(nnf_softplus(parameters$total_height + self$shift))
    total_height <- (self$default_height - self$min_height) * total_height +
      self$min_height

    bin_widths <- nnf_softmax(parameters$bin_widths, dim = -1L)
    bin_widths <- (total_width - self$bins * self$min_bin_width) * bin_widths +
      self$min_bin_width
    bin_heights <- nnf_softmax(parameters$bin_heights, dim = -1L)
    bin_heights <- (total_height - self$bins * self$min_bin_height) * bin_heights +
      self$min_bin_height

    affine_scale <- total_height / total_width
    affine_shift <- bottom_edge - affine_scale * left_edge

    horizontal_edges <- torch_cumsum(bin_widths, dim = -1L)
    horizontal_edges <- torch_cat(
      list(torch_zeros_like(left_edge), horizontal_edges),
      dim = -1L
    )
    horizontal_edges <- left_edge + horizontal_edges

    vertical_edges <- torch_cumsum(bin_heights, dim = -1L)
    vertical_edges <- torch_cat(
      list(torch_zeros_like(bottom_edge), vertical_edges),
      dim = -1L
    )
    vertical_edges <- bottom_edge + vertical_edges

    derivatives <- .torch_shifted_softplus(parameters$derivatives)
    derivatives <- torch_cat(
      list(affine_scale, derivatives, affine_scale),
      dim = -1L
    )

    list(
      horizontal_edges = horizontal_edges,
      vertical_edges = vertical_edges,
      derivatives = derivatives,
      affine_scale = torch_squeeze(affine_scale, -1L),
      affine_shift = torch_squeeze(affine_shift, -1L)
    )
  },
  transform_with_parameters = function(input, parameters, inverse = FALSE) {
    parameters <- self$constrain_parameters(self$split_parameters(parameters))

    scale <- parameters$affine_scale
    shift <- parameters$affine_shift
    if (inverse) {
      affine <- (input - shift) / scale
      affine_log_jacobian <- -torch_log(scale)
      edges_for_search <- parameters$vertical_edges
    } else {
      affine <- scale * input + shift
      affine_log_jacobian <- torch_log(scale)
      edges_for_search <- parameters$horizontal_edges
    }

    bins <- torch_squeeze(
      torch_searchsorted(edges_for_search, input$unsqueeze(-1L)),
      -1L
    )
    inside <- (bins > 0L) & (bins <= self$bins)

    upper <- torch_where(inside, bins, torch_ones_like(bins))
    lower <- torch_where(inside, bins - 1L, torch_zeros_like(bins))
    upper <- upper$unsqueeze(-1L)
    lower <- lower$unsqueeze(-1L)

    left <- torch_squeeze(
      .torch_spline_gather(parameters$horizontal_edges, lower),
      -1L
    )
    right <- torch_squeeze(
      .torch_spline_gather(parameters$horizontal_edges, upper),
      -1L
    )
    bottom <- torch_squeeze(
      .torch_spline_gather(parameters$vertical_edges, lower),
      -1L
    )
    top <- torch_squeeze(
      .torch_spline_gather(parameters$vertical_edges, upper),
      -1L
    )
    derivative_left <- torch_squeeze(
      .torch_spline_gather(parameters$derivatives, lower),
      -1L
    )
    derivative_right <- torch_squeeze(
      .torch_spline_gather(parameters$derivatives, upper),
      -1L
    )

    spline <- .torch_rational_quadratic_spline(
      input,
      left = left,
      right = right,
      bottom = bottom,
      top = top,
      derivative_left = derivative_left,
      derivative_right = derivative_right,
      inverse = inverse
    )

    output <- torch_where(inside, spline$output, affine)
    log_jacobian <- torch_where(
      inside,
      spline$log_jacobian,
      affine_log_jacobian
    )

    attr(output, "log_jacobian") <- torch_sum(log_jacobian, -1L, keepdim = TRUE)
    output
  },
  forward = function(input, parameters) {
    self$transform_with_parameters(input, parameters, inverse = FALSE)
  },
  reverse = function(input, parameters) {
    self$transform_with_parameters(input, parameters, inverse = TRUE)
  }
)
