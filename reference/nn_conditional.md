# Conditional Module

A conditional module is a module that takes an additional conditioning
input to the forward pass.

## Usage

``` r
nn_conditional()
```

## Forward method

The forward method should take two arguments, `input` and
`conditioning`, and return the output. Example:

    forward = function(input, conditioning) {
      output <- ...
      output
    }

## See also

[`nn_conditional_mlp()`](https://mbertolacci.github.io/torchflow/reference/nn_conditional_mlp.md)
for a concrete implementation.
