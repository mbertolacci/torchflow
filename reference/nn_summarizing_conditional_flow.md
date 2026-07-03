# Summarizing Conditional Flow

A summarizing conditional flow is a conditional flow that summarizes the
conditioning input using a summary model. It inherits from
[`nn_conditional_flow()`](https://mbertolacci.github.io/torchflow/reference/nn_conditional_flow.md),
and it requires a `summary_model` and a `flow_model` in its initializer.
During the forward pass, the summary model is applied to the
conditioning input, and the result is passed to the flow model. This can
be used to reduce the dimensionality of the conditioning input.

## Usage

``` r
nn_summarizing_conditional_flow(summary_model, flow_model)
```

## Arguments

- summary_model:

  A
  [`torch::nn_module()`](https://torch.mlverse.org/docs/reference/nn_module.html)
  that summarizes the conditioning input.

- flow_model:

  A
  [`torch::nn_module()`](https://torch.mlverse.org/docs/reference/nn_module.html)
  that is a conditional flow.

## Details

Its `forward()` method has an optional `summary` argument. If `summary`
is provided, it is used as the conditioning input for the flow model,
skipping the application of the summary model. This can be used to speed
up the forward pass if the same conditioning input is used multiple
times.

The class also provides a `summarize()` method that can be used to
compute the summary of the conditioning input outside of the forward
pass.

## See also

[`nn_conditional_flow()`](https://mbertolacci.github.io/torchflow/reference/nn_conditional_flow.md)

## Examples

``` r
library(torch)
summary_model <- nn_sequential(
  nn_linear(10, 5),
  nn_relu()
)
flow_model <- nn_affine_coupling_block(10, 5)
summarizing_flow <- nn_summarizing_conditional_flow(summary_model, flow_model)
```
