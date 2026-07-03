# Spline Coupling Transform

A rational-quadratic spline coupling transform following the BayesFlow
parameterization. The transform learns a monotone spline on a
rectangular domain and applies the affine map implied by that rectangle
outside the domain.

## Usage

``` r
nn_spline_coupling_transform(
  bins = 16,
  default_domain = c(-3, 3, -3, 3),
  min_width = 1,
  min_height = 1,
  min_bin_width = 0.1,
  min_bin_height = 0.1,
  method = "rational_quadratic"
)
```

## Arguments

- bins:

  The number of spline bins.

- default_domain:

  A numeric vector `c(left, right, bottom, top)` giving the default
  spline domain.

- min_width:

  The minimum total width of the learned domain.

- min_height:

  The minimum total height of the learned domain.

- min_bin_width:

  The minimum width of each bin.

- min_bin_height:

  The minimum height of each bin.

- method:

  The spline method. Currently only `"rational_quadratic"` is supported.

## Details

Its parameter tensor contains, for each transformed dimension, raw
values for the left edge, bottom edge, total width, total height, bin
widths, bin heights, and interior derivatives. With zero raw parameters,
the default domain gives an identity transform.

## Examples

``` r
library(torch)
transform <- nn_spline_coupling_transform(bins = 8)
transform$params_per_dim()
#> [1] 27
```
