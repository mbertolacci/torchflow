# Conditional Multilayer Perceptron

A conditional multilayer perceptron is a multilayer perceptron that
takes an additional conditioning input. It inherits from
[`nn_conditional()`](https://mbertolacci.github.io/torchflow/reference/nn_conditional.md);
it is not a normalizing flow. In practice, the regular input and the
conditioning input are concatenated and passed through the MLP.

## Usage

``` r
nn_conditional_mlp(
  input_size,
  conditioning_size,
  output_size,
  layer_sizes = c(128, 128),
  activation = nn_relu
)
```

## Arguments

- input_size:

  The size of the input to the MLP.

- conditioning_size:

  The size of the conditioning input to the MLP.

- output_size:

  The size of the output of the MLP.

- layer_sizes:

  A vector of integers specifying the number of neurons in each layer.
  This can be NULL, in which case a single linear layer is used.

- activation:

  The activation function to use after each layer.

## Examples

``` r
library(torch)
mlp <- nn_conditional_mlp(10, 5, 1)
input <- torch_randn(10)
conditioning <- torch_randn(5)
output <- mlp(input, conditioning)
```
