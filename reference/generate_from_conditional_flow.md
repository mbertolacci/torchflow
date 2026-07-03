# Generate samples from a conditional flow model

This function generates samples from a conditional flow model. If
conditioning is provided, `n_samples_per_batch` samples are generated
for each batch of the conditioning variable. If conditioning is not
provided, `n_samples_per_batch` samples are generated.

## Usage

``` r
generate_from_conditional_flow(model, n_samples_per_batch, conditioning)
```

## Arguments

- model:

  A conditional flow model.

- n_samples_per_batch:

  The number of samples to generate for each batch of the conditioning
  variable, or the total number of samples if conditioning is not
  provided.

- conditioning:

  The conditioning variable, a torch tensor of dimensions `[batch, ...]`
  where `batch` is the dimension of the batch and `...` are the
  dimensions of the conditioning variable.
