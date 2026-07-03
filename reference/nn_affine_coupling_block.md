# Affine Coupling Block

`nn_affine_coupling_block()` is a convenience constructor for an affine
coupling block. It constructs a dual coupling block when
`input_size > 1` and a single coupling block when `input_size = 1`. Use
[`nn_dual_coupling_block()`](https://mbertolacci.github.io/torchflow/reference/nn_dual_coupling_block.md)
or
[`nn_single_coupling_block()`](https://mbertolacci.github.io/torchflow/reference/nn_single_coupling_block.md)
directly to choose a different transform.

## Usage

``` r
nn_affine_coupling_block(
  input_size,
  conditioning_size = 0,
  left_size = if (input_size == 1L) 1L else as.integer(input_size%/%2),
  clamp = TRUE,
  ...
)
```

## Arguments

- input_size:

  The dimension of the input. The input itself is a tensor with
  dimensions `[batch_size, input_size]`, or just `[input_size]` if there
  is no batch dimension.

- conditioning_size:

  The dimension of the conditioning input, which has the same batch
  dimensions as the input.

- left_size:

  The dimension of the left part of the input (the split \\x_1\\ in the
  equations above).

- clamp:

  Whether to apply [`asinh()`](https://rdrr.io/r/base/Hyperbolic.html)
  to the raw scale before the shifted softplus constraint.

- ...:

  Additional arguments passed to the selected coupling block, such as
  `params` for univariate inputs or `f_params` and `g_params` for
  multivariate inputs.

## Details

An affine coupling block is a conditional flow inheriting from
[`nn_conditional_flow()`](https://mbertolacci.github.io/torchflow/reference/nn_conditional_flow.md)
that applies the following transformation to the input.

Let \\x = (x_1, x_2)\\ be a split of the input into two parts, and let
\\u\\ be the conditioning input. The forward transformation is given by:

\$\$ y_1 = x_1 s_f(x_2, u) + t_f(x_2, u) y_2 = x_2 s_g(y_1, u) +
t_g(y_1, u) \$\$

where the scales \\s_f\\ and \\s_g\\ are constrained to be positive by
the affine coupling transform.

By performing multiple such transformations in sequence, we can
construct a complex normalizing flow capable of modeling complicated
conditional distributions. Between each pair of such transformations,
the dimensions of the input should be permuted using a
[`nn_permutation_flow()`](https://mbertolacci.github.io/torchflow/reference/nn_permutation_flow.md).

When `input_size = 1`, this constructor warns because repeated
univariate affine coupling blocks compose to a single affine
transformation.

## Examples

``` r
library(torch)
# Coupling block used on its own with no conditioning
flow_model <- nn_affine_coupling_block(2, 0)
x <- torch_randn(10, 2)
y <- flow_model(x)
# y will be a tensor of dimensions [10, 2]
x_recovered <- flow_model$reverse(y)
# x_recovered will be a tensor of dimensions [10, 2]
# and numerically close to the original x

# Coupling block used with conditioning
flow_model <- nn_affine_coupling_block(2, 4)
x <- torch_randn(10, 2)
u <- torch_randn(10, 4)
y <- flow_model(x, u)
x_recovered <- flow_model$reverse(y, u)

# Coupling block used as part of a more complex flow model
flow_model <- nn_sequential_conditional_flow(
  nn_affine_coupling_block(2, 4),
  nn_permutation_flow(2),
  nn_affine_coupling_block(2, 4)
)
y <- flow_model(x, u)
```
