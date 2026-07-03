# ActNorm Block

An ActNorm block is a conditional flow inheriting from
[`nn_conditional_flow()`](https://mbertolacci.github.io/torchflow/reference/nn_conditional_flow.md)
that applies a learned per-dimension affine transformation,

## Usage

``` r
nn_actnorm_block(input_size)
```

## Arguments

- input_size:

  The size of the input to the flow.

## Details

\$\$y = s x + b\$\$

where `s` and `b` are trainable vectors initialized to one and zero. The
block ignores conditioning inputs and can be inserted directly into
[`nn_sequential_conditional_flow()`](https://mbertolacci.github.io/torchflow/reference/nn_sequential_conditional_flow.md).

## See also

[`nn_conditional_flow()`](https://mbertolacci.github.io/torchflow/reference/nn_conditional_flow.md)

## Examples

``` r
library(torch)
flow_model <- nn_sequential_conditional_flow(
  nn_actnorm_block(2),
  nn_spline_coupling_block(2),
  nn_permutation_flow(2),
  nn_actnorm_block(2),
  nn_spline_coupling_block(2)
)
```
