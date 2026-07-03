# Permutation Flow

A permutation flow is a conditional flow inheriting from
[`nn_conditional_flow()`](https://mbertolacci.github.io/torchflow/reference/nn_conditional_flow.md)
that permutes the input dimensions. The permutation is fixed to a random
permutation at initialization and does not change. It's log Jacobian is
zero since it is a simple reordering of the input dimensions. When
`input_size = 1`, the permutation is a no-op and a warning is issued.

## Usage

``` r
nn_permutation_flow(input_size)
```

## Arguments

- input_size:

  The size of the input to the flow.

## See also

[`nn_conditional_flow()`](https://mbertolacci.github.io/torchflow/reference/nn_conditional_flow.md)

## Examples

``` r
library(torch)
# Use on its own
permutation_flow <- nn_permutation_flow(10)
input <- torch_randn(10)
output <- permutation_flow(input)
# Use in a more complex conditional flow
flow_model <- nn_sequential_conditional_flow(
  nn_affine_coupling_block(10, 5),
  nn_permutation_flow(10),
  nn_affine_coupling_block(10, 5)
)
```
