# Single Coupling Block

A single coupling block splits the input into left and right parts. One
part is left unchanged and is used to compute the parameters for
transforming the other part with a coupling transform.

## Usage

``` r
nn_single_coupling_block(
  input_size,
  conditioning_size = 0,
  left_size = if (input_size == 1L) 1L else as.integer(input_size%/%2),
  transform = "affine",
  params,
  transform_left = TRUE,
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

- params:

  A conditional network returning the raw transform parameters.

- transform_left:

  Whether to transform the left part using parameters computed from the
  right part. If `FALSE`, the right part is transformed using parameters
  computed from the left part.

- ...:

  Additional arguments passed to the transform constructor.

## Examples

``` r
library(torch)
coupling <- nn_single_coupling_block(4, transform = "affine")
x <- torch_randn(10, 4)
y <- coupling(x)
x_recovered <- coupling$reverse(y)
```
