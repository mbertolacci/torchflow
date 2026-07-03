# Forward KL Loss

Compute the forward KL loss for a flow model.

## Usage

``` r
forward_kl_loss(input)
```

## Arguments

- input:

  The output of the flow model, which should be a tensor of dimensions
  `[batch1, ..., batchN, d]` where `batch1, ..., batchN` are the
  dimensions of the batch and `d` is the dimension of the input. It must
  also have an attribute `log_jacobian` containing the log determinant
  of the Jacobian of the transformation.

## Examples

``` r
library(torch)
flow_model <- nn_sequential_conditional_flow(
  nn_affine_coupling_block(2, 0),
  nn_permutation_flow(2),
  nn_affine_coupling_block(2, 0)
)
x <- torch_randn(10, 2)
y <- flow_model(x)
loss <- forward_kl_loss(y)
```
