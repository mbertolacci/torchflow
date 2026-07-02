.torch_head <- function(x, n) {
  if (n == 0L) {
    output_size <- x$size()
    output_size[length(output_size)] <- 0L
    return(torch_empty(output_size, device = x$device, dtype = x$dtype))
  }

  torch_index_select(
    x,
    -1,
    torch_tensor(1L : n, device = x$device)
  )
}

.torch_tail <- function(x, n) {
  if (n == 0L) {
    output_size <- x$size()
    output_size[length(output_size)] <- 0L
    return(torch_empty(output_size, device = x$device, dtype = x$dtype))
  }

  torch_index_select(
    x,
    -1,
    torch_tensor((x$size(-1) - n + 1L) : x$size(-1), device = x$device)
  )
}

.nn_raw_coupling_params <- nn_module(
  inherit = nn_conditional,
  initialize = function(output_size) {
    self$output_size <- as.integer(output_size)
    self$raw_parameters <- nn_parameter(torch_zeros(self$output_size))
  },
  forward = function(input, conditioning) {
    input_size <- input$size()
    output_size <- c(
      input_size[-length(input_size)],
      self$output_size
    )
    self$raw_parameters$expand(output_size)
  }
)

.nn_coupling_params <- nn_module(
  inherit = nn_conditional,
  initialize = function(
    input_size,
    conditioning_size,
    output_size,
    layer_sizes = c(128, 128),
    activation = nn_relu
  ) {
    sizes <- c(input_size + conditioning_size, layer_sizes)
    layers <- NULL
    if (length(sizes) > 1L) {
      for (i in 2 : length(sizes)) {
        layers <- c(layers, nn_linear(sizes[i - 1], sizes[i]))
        layers <- c(layers, activation())
      }
    }

    final_input_size <- sizes[length(sizes)]
    output_projector <- nn_linear(final_input_size, output_size)
    nn_init_zeros_(output_projector$weight)
    nn_init_zeros_(output_projector$bias)

    self$model <- if (is.null(layers)) {
      output_projector
    } else {
      do.call(nn_sequential, c(layers, output_projector))
    }
  },
  forward = function(input, conditioning) {
    if (!missing(conditioning)) {
      input <- torch_cat(list(input, conditioning), -1)
    }
    self$model(input)
  }
)

#' Single Coupling Block
#'
#' A single coupling block splits the input into left and right parts. One part
#' is left unchanged and is used to compute the parameters for transforming the
#' other part with a coupling transform.
#'
#' @param input_size The dimension of the input.
#' @param conditioning_size The dimension of the conditioning input.
#' @param left_size The dimension of the left part of the input.
#' @param transform The transform to apply. Currently `"affine"` is supported.
#'   A transform constructor or transform module can also be supplied.
#' @param params A conditional network returning the raw transform parameters.
#' @param transform_left Whether to transform the left part using parameters
#'   computed from the right part. If `FALSE`, the right part is transformed
#'   using parameters computed from the left part.
#' @param ... Additional arguments passed to the transform constructor.
#'
#' @examples
#' library(torch)
#' coupling <- nn_single_coupling_block(4, transform = "affine")
#' x <- torch_randn(10, 4)
#' y <- coupling(x)
#' x_recovered <- coupling$reverse(y)
#'
#' @export
nn_single_coupling_block <- nn_module(
  inherit = nn_conditional_flow,
  initialize = function(
    input_size,
    conditioning_size = 0,
    left_size = if (input_size == 1L) 1L else as.integer(input_size %/% 2),
    transform = "affine",
    params,
    transform_left = TRUE,
    ...
  ) {
    self$input_size <- input_size
    self$left_size <- left_size
    self$right_size <- input_size - left_size
    self$transform_left <- transform_left
    self$transform <- .resolve_coupling_transform(transform, ...)

    if (transform_left) {
      params_input_size <- self$right_size
      params_output_size <- self$left_size
    } else {
      params_input_size <- self$left_size
      params_output_size <- self$right_size
    }

    self$params <- if (missing(params) && params_input_size == 0L && conditioning_size == 0L) {
      .nn_raw_coupling_params(
        self$transform$params_per_dim() * params_output_size
      )
    } else if (missing(params)) {
      .nn_coupling_params(
        params_input_size,
        conditioning_size,
        self$transform$params_per_dim() * params_output_size
      )
    } else {
      params
    }
  },
  forward = function(input, conditioning) {
    input1 <- .torch_head(input, self$left_size)
    input2 <- .torch_tail(input, self$right_size)

    if (self$transform_left) {
      output1 <- self$transform(input1, self$params(input2, conditioning))
      output2 <- input2
    } else {
      output1 <- input1
      output2 <- self$transform(input2, self$params(input1, conditioning))
    }

    output <- torch_cat(list(output1, output2), -1)
    attr(output, "log_jacobian") <- if (self$transform_left) {
      attr(output1, "log_jacobian")
    } else {
      attr(output2, "log_jacobian")
    }
    output
  },
  reverse = function(input, conditioning) {
    input1 <- .torch_head(input, self$left_size)
    input2 <- .torch_tail(input, self$right_size)

    if (self$transform_left) {
      output1 <- self$transform$reverse(input1, self$params(input2, conditioning))
      output2 <- input2
    } else {
      output1 <- input1
      output2 <- self$transform$reverse(input2, self$params(input1, conditioning))
    }

    torch_cat(list(output1, output2), -1)
  },
  dimension = function() {
    self$input_size
  }
)

#' Dual Coupling Block
#'
#' A dual coupling block applies two single coupling transformations in
#' sequence: first transforming the left part from the right part, then
#' transforming the right part from the transformed left part. It requires
#' `input_size` greater than one.
#'
#' @param input_size The dimension of the input.
#' @param conditioning_size The dimension of the conditioning input.
#' @param left_size The dimension of the left part of the input.
#' @param transform The transform to apply. Currently `"affine"` is supported.
#'   A transform constructor or transform module can also be supplied.
#' @param f_params A conditional network returning parameters for transforming
#'   the left part from the right part.
#' @param g_params A conditional network returning parameters for transforming
#'   the right part from the transformed left part.
#' @param ... Additional arguments passed to the transform constructor.
#'
#' @examples
#' library(torch)
#' coupling <- nn_dual_coupling_block(4, transform = "affine")
#' x <- torch_randn(10, 4)
#' y <- coupling(x)
#' x_recovered <- coupling$reverse(y)
#'
#' @export
nn_dual_coupling_block <- nn_module(
  inherit = nn_conditional_flow,
  initialize = function(
    input_size,
    conditioning_size = 0,
    left_size = as.integer(input_size %/% 2),
    transform = "affine",
    f_params,
    g_params,
    ...
  ) {
    if (input_size == 1L) {
      stop(
        "`nn_dual_coupling_block()` requires `input_size` greater than 1.",
        call. = FALSE
      )
    }

    self$input_size <- input_size
    self$left_size <- left_size

    self$f_coupling <- if (missing(f_params)) {
      nn_single_coupling_block(
        input_size,
        conditioning_size,
        left_size,
        transform = transform,
        transform_left = TRUE,
        ...
      )
    } else {
      nn_single_coupling_block(
        input_size,
        conditioning_size,
        left_size,
        transform = transform,
        params = f_params,
        transform_left = TRUE,
        ...
      )
    }

    self$g_coupling <- if (missing(g_params)) {
      nn_single_coupling_block(
        input_size,
        conditioning_size,
        left_size,
        transform = transform,
        transform_left = FALSE,
        ...
      )
    } else {
      nn_single_coupling_block(
        input_size,
        conditioning_size,
        left_size,
        transform = transform,
        params = g_params,
        transform_left = FALSE,
        ...
      )
    }
  },
  forward = function(input, conditioning) {
    output <- self$f_coupling(input, conditioning)
    log_jacobian_f <- attr(output, "log_jacobian")

    output <- self$g_coupling(output, conditioning)
    log_jacobian_g <- attr(output, "log_jacobian")
    attr(output, "log_jacobian") <- log_jacobian_f + log_jacobian_g
    output
  },
  reverse = function(input, conditioning) {
    output <- self$g_coupling$reverse(input, conditioning)
    self$f_coupling$reverse(output, conditioning)
  },
  dimension = function() {
    self$input_size
  }
)

#' Affine Coupling Block
#'
#' `nn_affine_coupling_block()` is a convenience constructor for an affine
#' coupling block. It constructs a dual coupling block when `input_size > 1`
#' and a single coupling block when `input_size = 1`. Use
#' [nn_dual_coupling_block()] or [nn_single_coupling_block()] directly to choose
#' a different transform.
#'
#' An affine coupling block is a conditional flow inheriting from
#' [nn_conditional_flow()] that applies the following transformation to the
#' input.
#'
#' Let \eqn{x = (x_1, x_2)} be a split of the input into two parts, and let
#' \eqn{u} be the conditioning input. The forward transformation is given by:
#'
#' \deqn{
#'   y_1 = x_1 s_f(x_2, u) + t_f(x_2, u)
#'   y_2 = x_2 s_g(y_1, u) + t_g(y_1, u)
#' }
#'
#' where the scales \eqn{s_f} and \eqn{s_g} are constrained to be positive by
#' the affine coupling transform.
#'
#' By performing multiple such transformations in sequence, we can construct a
#' complex normalizing flow capable of modeling complicated conditional
#' distributions. Between each pair of such transformations, the dimensions of
#' the input should be permuted using a [nn_permutation_flow()].
#'
#' When `input_size = 1`, this constructor warns because repeated univariate
#' affine coupling blocks compose to a single affine transformation.
#'
#' @param input_size The dimension of the input. The input itself is a tensor
#' with dimensions `[batch_size, input_size]`, or just `[input_size]` if there
#' is no batch dimension.
#' @param conditioning_size The dimension of the conditioning input, which has
#' the same batch dimensions as the input.
#' @param left_size The dimension of the left part of the input (the split
#'   \eqn{x_1} in the equations above).
#' @param clamp Whether to apply `asinh()` to the raw scale before the shifted
#'   softplus constraint.
#' @param ... Additional arguments passed to the selected coupling block, such as
#'   `params` for univariate inputs or `f_params` and `g_params` for
#'   multivariate inputs.
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
#' x_recovered <- flow_model$reverse(y, u)
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
    left_size = if (input_size == 1L) 1L else as.integer(input_size %/% 2),
    clamp = TRUE,
    ...
  ) {
    dots <- list(...)
    if ("transform" %in% names(dots)) {
      stop(
        "`nn_affine_coupling_block()` always uses `transform = \"affine\"`.",
        call. = FALSE
      )
    }

    if (input_size == 1L) {
      warning(
        "`nn_affine_coupling_block()` with `input_size = 1` uses a single affine ",
        "coupling; composing affine blocks is no better than a single affine block.",
        call. = FALSE
      )
    }

    coupling_block <- if (input_size == 1L) {
      nn_single_coupling_block
    } else {
      nn_dual_coupling_block
    }

    self$block <- do.call(
      coupling_block,
      c(
        list(
          input_size = input_size,
          conditioning_size = conditioning_size,
          left_size = left_size,
          clamp = clamp,
          transform = "affine"
        ),
        dots
      )
    )
  },
  forward = function(input, conditioning) {
    self$block(input, conditioning)
  },
  reverse = function(input, conditioning) {
    self$block$reverse(input, conditioning)
  },
  dimension = function() {
    self$block$dimension()
  }
)
