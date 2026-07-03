# Sequential Conditional Flow

A sequential conditional flow is a conditional flow inheriting from
[`nn_conditional_flow()`](https://mbertolacci.github.io/torchflow/reference/nn_conditional_flow.md)
that applies a sequence of conditional flows, passing the conditioning
input to each flow. It is the analog of
[`torch::nn_sequential()`](https://torch.mlverse.org/docs/reference/nn_sequential.html)
for conditional flows.

## Usage

``` r
nn_sequential_conditional_flow(...)
```

## Arguments

- ...:

  A sequence of conditional flows, or a list of conditional flows.

## See also

[`nn_conditional_flow()`](https://mbertolacci.github.io/torchflow/reference/nn_conditional_flow.md)

## Examples

``` r
flow_model <- nn_sequential_conditional_flow(
  nn_affine_coupling_block(10, 5),
  nn_permutation_flow(10),
  nn_affine_coupling_block(10, 5)
)
```
