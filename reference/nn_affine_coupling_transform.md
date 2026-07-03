# Affine Coupling Transform

An affine coupling transform applies `output = input * scale + shift`.
Its parameter tensor contains the unconstrained scale and shift values
concatenated along the last dimension. The scale is constrained to be
positive with a shifted softplus, so a raw scale of zero maps to a
multiplicative scale of one.

## Usage

``` r
nn_affine_coupling_transform(clamp = TRUE)
```

## Arguments

- clamp:

  Whether to apply [`asinh()`](https://rdrr.io/r/base/Hyperbolic.html)
  to the raw scale before the shifted softplus constraint.

## Examples

``` r
library(torch)
transform <- nn_affine_coupling_transform()
transform$params_per_dim()
#> [1] 2
```
