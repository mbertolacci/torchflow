# Dual Coupling Block

A dual coupling block applies two single coupling transformations in
sequence: first transforming the left part from the right part, then
transforming the right part from the transformed left part. It requires
`input_size` greater than one.

## Usage

``` r
nn_dual_coupling_block(
  input_size,
  conditioning_size = 0,
  left_size = as.integer(input_size%/%2),
  transform = "affine",
  f_params,
  g_params,
  ...
)
```

## Arguments

- input_size:

  The dimension of the input.

- conditioning_size:

  The dimension of the conditioning input.

- left_size:

  The dimension of the left part of the input.

- transform:

  The transform to apply. Currently `"affine"` and `"spline"` are
  supported. A transform constructor or transform module can also be
  supplied.

- f_params:

  A conditional network returning parameters for transforming the left
  part from the right part.

- g_params:

  A conditional network returning parameters for transforming the right
  part from the transformed left part.

- ...:

  Additional arguments passed to the transform constructor.

## Examples

``` r
library(torch)
coupling <- nn_dual_coupling_block(4, transform = "affine")
x <- torch_randn(10, 4)
y <- coupling(x)
x_recovered <- coupling$reverse(y)

spline_coupling <- nn_dual_coupling_block(4, transform = "spline")
```
