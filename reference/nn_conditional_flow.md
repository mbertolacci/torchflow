# Conditional Normalizing Flow

A conditional normalizing flow is a normalizing flow that takes an
additional conditioning input. This module provides a base class for
conditional normalizing flows.

## Usage

``` r
nn_conditional_flow()
```

## Details

The base class, `nn_conditional_flow`, is an abstract class that
provides a forward and reverse method, as well as a dimension method.
Subclasses created with
[`torch::nn_module()`](https://torch.mlverse.org/docs/reference/nn_module.html)
should implement these methods. The class is a subclass of
[`torch::nn_module()`](https://torch.mlverse.org/docs/reference/nn_module.html),
and it inherits all of its methods and semantics.

## Forward method

The forward method should return the output and the log determinant of
the Jacobian in the attribute `log_jacobian`. Example:

    forward = function(input, conditioning) {
      output <- ...
      attr(output, 'log_jacobian') <- ...
      output
    }

## Reverse method

The reverse method should return the inverse of the output, but need not
implement a log determinant. Example:

    reverse = function(input, conditioning) {
      output <- ...
      output
    }

## Dimension method

The dimension method should return the dimension of the input and output
of the flow. Example:

    dimension = function() {
      return(2)
    }

## See also

[`nn_summarizing_conditional_flow()`](https://mbertolacci.github.io/torchflow/reference/nn_summarizing_conditional_flow.md),
[`nn_sequential_conditional_flow()`](https://mbertolacci.github.io/torchflow/reference/nn_sequential_conditional_flow.md),
[`nn_permutation_flow()`](https://mbertolacci.github.io/torchflow/reference/nn_permutation_flow.md),
and
[`nn_affine_coupling_block()`](https://mbertolacci.github.io/torchflow/reference/nn_affine_coupling_block.md)
for subclasses.
