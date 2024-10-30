#' Conditional Module
#'
#' A conditional module is a module that takes an additional conditioning input to
#' the forward pass.
#'
#' @section Forward method:
#' The forward method should take two arguments, `input` and `conditioning`, and
#' return the output. Example:
#'
#' ```r
#' forward = function(input, conditioning) {
#'   output <- ...
#'   output
#' }
#' ```
#'
#' @seealso [nn_conditional_mlp()] for a concrete implementation.
#' @export
nn_conditional <- nn_module(
  forward = function(input, conditioning) {
    stop('Not implemented')
  }
)

#' Conditional Multilayer Perceptron
#'
#' A conditional multilayer perceptron is a multilayer perceptron that takes an
#' additional conditioning input. It inherits from [nn_conditional()]; it is
#' not a normalizing flow. In practice, the regular input and the conditioning
#' input are concatenated and passed through the MLP.
#'
#' @param input_size The size of the input to the MLP.
#' @param conditioning_size The size of the conditioning input to the MLP.
#' @param output_size The size of the output of the MLP.
#' @param layer_sizes A vector of integers specifying the number of neurons in each
#' layer. This can be NULL, in which case a single linear layer is used.
#' @param activation The activation function to use after each layer.
#'
#' @examples
#' library(torch)
#' mlp <- nn_conditional_mlp(10, 5, 1)
#' input <- torch_randn(10)
#' conditioning <- torch_randn(5)
#' output <- mlp(input, conditioning)
#' @export
nn_conditional_mlp <- nn_module(
  inherit = nn_conditional,
  initialize = function(
    input_size,
    conditioning_size,
    output_size,
    layer_sizes = c(128, 128),
    activation = nn_relu
  ) {
    sizes <- c(input_size + conditioning_size, layer_sizes, output_size)
    layers <- NULL
    for (i in 2 : length(sizes)) {
      layers <- c(layers, nn_linear(sizes[i - 1], sizes[i]))
      if (i < length(sizes)) {
        layers <- c(layers, activation())
      }
    }
    self$model <- do.call(nn_sequential, layers)
  },
  forward = function(input, conditioning) {
    if (!missing(conditioning)) {
      input <- torch_cat(list(input, conditioning), -1)
    }
    self$model(input)
  }
)
