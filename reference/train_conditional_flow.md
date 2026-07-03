# Train a conditional flow model

Method to train a conditional flow model. This is a basic training loop
with the following steps:

## Usage

``` r
train_conditional_flow(
  model,
  generate,
  optimizer = torch::optim_adam,
  n_epochs = 128,
  batch_size = 32,
  after_epoch = NULL,
  verbose = TRUE,
  ...
)
```

## Arguments

- model:

  A conditional flow model inheriting from
  [`nn_conditional_flow()`](https://mbertolacci.github.io/torchflow/reference/nn_conditional_flow.md).

- generate:

  A function that generates a batch of target and conditioning samples;
  see above for details. This will be passed the current epoch number as
  an argument.

- optimizer:

  The optimizer to use, e.g.
  [`torch::optim_adam()`](https://torch.mlverse.org/docs/reference/optim_adam.html).

- n_epochs:

  The number of epochs to train for.

- batch_size:

  The batch size.

- after_epoch:

  A function to call after each epoch.

- verbose:

  Whether to print progress.

- ...:

  Additional arguments to pass to the optimizer.

## Details

The training algorithm is as follows. For each epoch:

1.  Generate (using `generate`) a batch of target and conditioning
    samples.

2.  Loop over the batches of the epoch, performing a gradient descent
    step for each batch. The batches are processed in order from the
    generated samples.

3.  Call `after_epoch` (if provided) with the current epoch and the
    generated samples. This can be used to print test loss or any other
    tasks.

The `generate` function (called with the current epoch number as an
argument) should return a list with the following elements:

- `target`: An [`array()`](https://rdrr.io/r/base/array.html),
  [`matrix()`](https://rdrr.io/r/base/matrix.html), or
  [`torch::torch_tensor()`](https://torch.mlverse.org/docs/reference/torch_tensor.html)
  of target samples.

- `conditioning`: An optional
  [`array()`](https://rdrr.io/r/base/array.html),
  [`matrix()`](https://rdrr.io/r/base/matrix.html), or
  [`torch::torch_tensor()`](https://torch.mlverse.org/docs/reference/torch_tensor.html)
  of conditioning samples. If returning
  [`torch_tensor()`](https://torch.mlverse.org/docs/reference/torch_tensor.html)
  objects, take care that they are on the same device as the model.

The generated samples are the choice of the user. You could generate new
samples each epoch, or share the same samples across epochs (noting that
the model may overfit in this case). In that latter case, it would be
good to permute the order of the samples each epoch.

The training may be stopped early. The original model object is modified
in place.

## Examples

``` r
library(torch)
model <- nn_sequential_conditional_flow(
  nn_affine_coupling_block(input_size = 2),
  nn_permutation_flow(input_size = 2),
  nn_affine_coupling_block(input_size = 2)
)
generate <- function(epoch) {
  list(target = 2 + torch_randn(1024, 2))
}
# In practice, the number of epochs should be larger
train_conditional_flow(model, generate, n_epochs = 2)
#> = Starting epoch 1 
#> = Starting epoch 2 
```
