# Coupling Transform

A coupling transform maps an input partition and a set of parameters to
an invertible output partition. Coupling blocks use `params_per_dim()`
to size their parameter networks.

## Usage

``` r
nn_coupling_transform()
```

## Transform methods

Subclasses should implement `params_per_dim()`, `split_parameters()`,
`constrain_parameters()`, `forward()`, and `reverse()`.
