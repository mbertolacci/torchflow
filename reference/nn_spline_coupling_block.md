# Spline Coupling Block

`nn_spline_coupling_block()` is a convenience constructor for a
rational-quadratic spline coupling block. It constructs a dual coupling
block when `input_size > 1` and a single coupling block when
`input_size = 1`. Use
[`nn_dual_coupling_block()`](https://mbertolacci.github.io/torchflow/reference/nn_dual_coupling_block.md)
or
[`nn_single_coupling_block()`](https://mbertolacci.github.io/torchflow/reference/nn_single_coupling_block.md)
directly to choose a different transform.

## Usage

``` r
nn_spline_coupling_block(
  input_size,
  conditioning_size = 0,
  left_size = if (input_size == 1L) 1L else as.integer(input_size%/%2),
  ...
)
```

## Arguments

- input_size:

  The dimension of the input. The input itself is a tensor with
  dimensions `[batch_size, input_size]`, or just `[input_size]` if there
  is no batch dimension.

- conditioning_size:

  The dimension of the conditioning input, which has the same batch
  dimensions as the input.

- left_size:

  The dimension of the left part of the input.

- ...:

  Additional arguments passed to the selected coupling block and spline
  transform, such as `bins`, `params` for univariate inputs, or
  `f_params` and `g_params` for multivariate inputs.

## Examples

``` r
library(torch)
flow_model <- nn_spline_coupling_block(2, bins = 8)
x <- torch_randn(10, 2)
y <- flow_model(x)
x_recovered <- flow_model$reverse(y)
```
