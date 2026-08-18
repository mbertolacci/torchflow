# torchflow

``` r

library(torch)
library(torchflow)
```

## Introduction

*This vignette is still under construction*

**Give mathematical details of normalizing flows**

## Creating and sampling from a flow

A simple two parameter normalising flow can be created as follows:

``` r

flow_model <- nn_sequential_conditional_flow(
  nn_affine_coupling_block(2),
  nn_permutation_flow(2),
  nn_affine_coupling_block(2),
  nn_permutation_flow(2),
  nn_affine_coupling_block(2)
)
```

This flow has five layers, which alternate between affine coupling
blocks and permutation flows. The flow is just a standard torch
`nn_module` and can be used in the usual way:

``` r

x <- torch_randn(5, 2)
flow_model(x)
#> torch_tensor
#> -1.7643 -0.6731
#>  0.1615  0.8262
#> -1.6197  0.3996
#>  1.4587 -0.0257
#> -2.3079 -0.1036
#> [ CPUFloatType{5,2} ][ grad_fn = <CatBackward0> ]
```

The first dimension of the input acts as a batch dimension, so the flow
can be used to generate multiple samples at once. The above code
actually implements sampling from the distribution represented by the
flow. This can also be done directly using the
`generate_from_conditional_flow` function:

``` r

generate_from_conditional_flow(flow_model, 5)
#> torch_tensor
#>  0.1446 -0.3376
#> -1.4231  0.1548
#>  0.9433  0.6673
#>  0.4506 -1.5292
#> -0.6118 -0.6629
#> [ CPUFloatType{5,2} ][ grad_fn = <ViewBackward0> ]
```

## Spline coupling flows

Coupling blocks can also use rational-quadratic spline transforms.
ActNorm blocks can be inserted as standalone layers:

``` r

spline_flow_model <- nn_sequential_conditional_flow(
  nn_actnorm_block(2),
  nn_spline_coupling_block(2, bins = 8),
  nn_permutation_flow(2),
  nn_actnorm_block(2),
  nn_spline_coupling_block(2, bins = 8)
)

x <- torch_randn(5, 2)
y <- spline_flow_model(x)
x_recovered <- spline_flow_model$reverse(y)

y
#> torch_tensor
#> -1.5688  0.3759
#>  1.6414  0.5290
#>  0.4800  1.1087
#> -0.7837 -0.4561
#>  1.4390  0.5675
#> [ CPUFloatType{5,2} ][ grad_fn = <CatBackward0> ]
torch_max(torch_abs(x - x_recovered))
#> torch_tensor
#> 2.9802322387695312e-08
#> [ CPUFloatType{} ][ grad_fn = <MaxBackward1> ]
```

## Univariate flows

The same interface can be used for a univariate target by setting
`input_size = 1`. Spline coupling blocks use a single coupling
internally, and can be composed to build richer univariate flows.

``` r

flow_model <- nn_sequential_conditional_flow(
  nn_spline_coupling_block(1),
  nn_spline_coupling_block(1)
)

x <- torch_randn(5, 1)
y <- flow_model(x)
x_recovered <- flow_model$reverse(y)

y
#> torch_tensor
#>  0.5577
#>  0.4343
#> -1.8791
#>  0.4799
#>  2.1265
#> [ CPUFloatType{5,1} ][ grad_fn = <CatBackward0> ]
torch_max(torch_abs(x - x_recovered))
#> torch_tensor
#> 0
#> [ CPUFloatType{} ][ grad_fn = <MaxBackward1> ]
```

Sampling works in the same way as for multivariate flows:

``` r

generate_from_conditional_flow(flow_model, 5)
#> torch_tensor
#> -0.8748
#>  0.9877
#>  2.3516
#> -0.9081
#>  0.2622
#> [ CPUFloatType{5,1} ][ grad_fn = <ViewBackward0> ]
```

## Conditional flow

A conditional flow takes an additional input, the conditioning variable,
which can be used to condition the samples on some additional
information. The flow therefore encodes a conditional distribution. The
following code creates a conditional flow with the same architecture as
the unconditional flow defined above but with an additional conditioning
variable of dimension 3:

``` r

flow_model <- nn_sequential_conditional_flow(
  nn_affine_coupling_block(2, 3),
  nn_permutation_flow(2),
  nn_affine_coupling_block(2, 3),
  nn_permutation_flow(2),
  nn_affine_coupling_block(2, 3)
)
```

We can sample from the flow for a given conditioning variable as
follows:

``` r

conditioning <- torch_randn(3)
generate_from_conditional_flow(flow_model, 5, conditioning)
#> torch_tensor
#>  1.3037  0.5725
#>  0.7520  2.5229
#> -0.1383  0.4189
#> -0.8427  1.3874
#>  0.9764  0.7915
#> [ CPUFloatType{5,2} ][ grad_fn = <ViewBackward0> ]
```

We can also do this for a batch of conditioning variables:

``` r

conditioning <- torch_randn(8, 3)
generate_from_conditional_flow(flow_model, 5, conditioning)
#> torch_tensor
#> (1,.,.) = 
#>  0.2580  0.9484
#>   1.1710 -1.4612
#>   0.5786 -1.7176
#>   0.4977  0.2994
#>  -0.1501  1.7049
#>   1.1069  0.3332
#>  -0.2313  0.1653
#>  -1.5135  0.2784
#> 
#> (2,.,.) = 
#> -2.0329  1.4437
#>   0.3373  0.5657
#>   1.2622  1.4830
#>   0.0952 -0.4900
#>  -0.6994 -1.6108
#>   0.0736  0.3558
#>  -0.0750 -0.5050
#>  -1.1546 -0.8480
#> 
#> (3,.,.) = 
#>  0.0632  0.0347
#>  -0.2204  0.5619
#>  -0.7296 -0.7859
#>  -0.1002 -0.5365
#>  -0.2507 -1.1429
#>  -0.8594 -1.4960
#>  -1.1675 -0.9385
#>   0.8372 -0.1389
#> 
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{5,8,2} ][ grad_fn = <ViewBackward0> ]
```

## Training an unconditional flow

The above flows are randomly initialised, so samples from them do not
follow any interesting distribution. We can instead train a flow to
follow a given distribution using samples from that distribution.

Let us train a flow to match the following distribution
$`\sigma \sim N^+(0, 1)`$, $`\mu \sim N(0, \sigma^2)`$, where
$`N^+(0, 1)`$ is the half normal distribution with mean 0 and standard
deviation 1. We can generate samples from this distribution as follows.
Note that we take the logarithm of the $`\sigma`$ parameter to ensure
that it has real support; the flow targets this transformed
distribution.

``` r

generate_samples <- function(...) {
  n_samples <- 1024
  sigma <- torch_abs(torch_randn(n_samples))
  mu <- torch_randn(n_samples) * sigma
  list(target = torch_stack(list(mu, torch_log(sigma)), 2))
}

generate_samples()
#> $target
#> torch_tensor
#> -0.3237 -0.6801
#>  0.4244 -0.8948
#> -0.4852 -1.4182
#> -0.0800 -0.8686
#> -0.4226 -1.1163
#> -3.8252  0.5364
#> -0.6037  0.5260
#>  0.2892 -1.1443
#> -0.2108 -0.4647
#>  0.2437 -1.6185
#>  0.4543 -0.1250
#> -0.7346 -0.3445
#>  0.3688 -0.5011
#> -0.5484 -1.6510
#>  0.1959  0.1914
#>  1.6139  0.0179
#>  0.0180 -1.1707
#> -0.7871 -0.2547
#> -0.4208  0.1344
#> -4.6178  0.7348
#>  2.5131  0.9543
#> -0.0154 -2.3613
#>  0.0491 -3.2933
#>  0.3581  0.0349
#>  1.1786  0.4837
#> -0.1521 -0.0551
#>  0.6819 -0.0581
#> -0.5764 -1.1971
#>  0.0048 -2.8107
#>  1.1686 -0.2456
#> ... [the output was truncated (use n=-1 to disable)]
#> [ CPUFloatType{1024,2} ]
```

This function can be used to train the flow to match the distribution
using the `train_conditional_flow` function:

``` r

# Make an unconditional flow
flow_model <- nn_sequential_conditional_flow(
  nn_affine_coupling_block(2),
  nn_permutation_flow(2),
  nn_affine_coupling_block(2),
  nn_permutation_flow(2),
  nn_affine_coupling_block(2)
)

# Generate a test set
test_set <- generate_samples()

# Train the flow
train_conditional_flow(
  flow_model,
  generate_samples,
  n_epochs = 128,
  batch_size = 1024,
  after_epoch = function(...) {
    test_loss <- as_array(forward_kl_loss(flow_model(test_set$target)))
    cat('Test loss:', test_loss, '\n')
  }
)
#> = Starting epoch 1 
#> Test loss: 1.352987 
#> = Starting epoch 2 
#> Test loss: 1.278748 
#> = Starting epoch 3 
#> Test loss: 1.197736 
#> = Starting epoch 4 
#> Test loss: 1.113483 
#> = Starting epoch 5 
#> Test loss: 1.028094 
#> = Starting epoch 6 
#> Test loss: 0.9471374 
#> = Starting epoch 7 
#> Test loss: 0.8731999 
#> = Starting epoch 8 
#> Test loss: 0.7978641 
#> = Starting epoch 9 
#> Test loss: 0.729042 
#> = Starting epoch 10 
#> Test loss: 0.65716 
#> = Starting epoch 11 
#> Test loss: 0.6047417 
#> = Starting epoch 12 
#> Test loss: 0.5570822 
#> = Starting epoch 13 
#> Test loss: 0.5555143 
#> = Starting epoch 14 
#> Test loss: 0.5512314 
#> = Starting epoch 15 
#> Test loss: 0.5672613 
#> = Starting epoch 16 
#> Test loss: 0.5760901 
#> = Starting epoch 17 
#> Test loss: 0.5909589 
#> = Starting epoch 18 
#> Test loss: 0.5892233 
#> = Starting epoch 19 
#> Test loss: 0.574774 
#> = Starting epoch 20 
#> Test loss: 0.5740153 
#> = Starting epoch 21 
#> Test loss: 0.5349825 
#> = Starting epoch 22 
#> Test loss: 0.5036096 
#> = Starting epoch 23 
#> Test loss: 0.4956441 
#> = Starting epoch 24 
#> Test loss: 0.4813409 
#> = Starting epoch 25 
#> Test loss: 0.4654467 
#> = Starting epoch 26 
#> Test loss: 0.4600905 
#> = Starting epoch 27 
#> Test loss: 0.4659606 
#> = Starting epoch 28 
#> Test loss: 0.4659551 
#> = Starting epoch 29 
#> Test loss: 0.4651598 
#> = Starting epoch 30 
#> Test loss: 0.4665864 
#> = Starting epoch 31 
#> Test loss: 0.4657956 
#> = Starting epoch 32 
#> Test loss: 0.4616663 
#> = Starting epoch 33 
#> Test loss: 0.4585747 
#> = Starting epoch 34 
#> Test loss: 0.4562497 
#> = Starting epoch 35 
#> Test loss: 0.4499168 
#> = Starting epoch 36 
#> Test loss: 0.4424081 
#> = Starting epoch 37 
#> Test loss: 0.4392762 
#> = Starting epoch 38 
#> Test loss: 0.4315724 
#> = Starting epoch 39 
#> Test loss: 0.4296353 
#> = Starting epoch 40 
#> Test loss: 0.4308472 
#> = Starting epoch 41 
#> Test loss: 0.4270186 
#> = Starting epoch 42 
#> Test loss: 0.4230314 
#> = Starting epoch 43 
#> Test loss: 0.4238397 
#> = Starting epoch 44 
#> Test loss: 0.4201143 
#> = Starting epoch 45 
#> Test loss: 0.4218487 
#> = Starting epoch 46 
#> Test loss: 0.4182592 
#> = Starting epoch 47 
#> Test loss: 0.4134986 
#> = Starting epoch 48 
#> Test loss: 0.4163702 
#> = Starting epoch 49 
#> Test loss: 0.4091715 
#> = Starting epoch 50 
#> Test loss: 0.4177631 
#> = Starting epoch 51 
#> Test loss: 0.4136117 
#> = Starting epoch 52 
#> Test loss: 0.4066332 
#> = Starting epoch 53 
#> Test loss: 0.4172759 
#> = Starting epoch 54 
#> Test loss: 0.4045648 
#> = Starting epoch 55 
#> Test loss: 0.4140955 
#> = Starting epoch 56 
#> Test loss: 0.4134179 
#> = Starting epoch 57 
#> Test loss: 0.4041783 
#> = Starting epoch 58 
#> Test loss: 0.4099116 
#> = Starting epoch 59 
#> Test loss: 0.3985088 
#> = Starting epoch 60 
#> Test loss: 0.40016 
#> = Starting epoch 61 
#> Test loss: 0.3965876 
#> = Starting epoch 62 
#> Test loss: 0.3958271 
#> = Starting epoch 63 
#> Test loss: 0.3987118 
#> = Starting epoch 64 
#> Test loss: 0.398964 
#> = Starting epoch 65 
#> Test loss: 0.4008296 
#> = Starting epoch 66 
#> Test loss: 0.4017216 
#> = Starting epoch 67 
#> Test loss: 0.4012151 
#> = Starting epoch 68 
#> Test loss: 0.4031049 
#> = Starting epoch 69 
#> Test loss: 0.3998488 
#> = Starting epoch 70 
#> Test loss: 0.3955206 
#> = Starting epoch 71 
#> Test loss: 0.3913098 
#> = Starting epoch 72 
#> Test loss: 0.3883861 
#> = Starting epoch 73 
#> Test loss: 0.3853782 
#> = Starting epoch 74 
#> Test loss: 0.3871993 
#> = Starting epoch 75 
#> Test loss: 0.3942252 
#> = Starting epoch 76 
#> Test loss: 0.3828428 
#> = Starting epoch 77 
#> Test loss: 0.3858472 
#> = Starting epoch 78 
#> Test loss: 0.382165 
#> = Starting epoch 79 
#> Test loss: 0.3875421 
#> = Starting epoch 80 
#> Test loss: 0.3897325 
#> = Starting epoch 81 
#> Test loss: 0.3828738 
#> = Starting epoch 82 
#> Test loss: 0.3839411 
#> = Starting epoch 83 
#> Test loss: 0.3872853 
#> = Starting epoch 84 
#> Test loss: 0.3802991 
#> = Starting epoch 85 
#> Test loss: 0.3857347 
#> = Starting epoch 86 
#> Test loss: 0.3822364 
#> = Starting epoch 87 
#> Test loss: 0.3784367 
#> = Starting epoch 88 
#> Test loss: 0.3804981 
#> = Starting epoch 89 
#> Test loss: 0.3771877 
#> = Starting epoch 90 
#> Test loss: 0.3821052 
#> = Starting epoch 91 
#> Test loss: 0.3896319 
#> = Starting epoch 92 
#> Test loss: 0.3759111 
#> = Starting epoch 93 
#> Test loss: 0.3800206 
#> = Starting epoch 94 
#> Test loss: 0.3795123 
#> = Starting epoch 95 
#> Test loss: 0.3885282 
#> = Starting epoch 96 
#> Test loss: 0.4018164 
#> = Starting epoch 97 
#> Test loss: 0.3725834 
#> = Starting epoch 98 
#> Test loss: 0.3799687 
#> = Starting epoch 99 
#> Test loss: 0.3781576 
#> = Starting epoch 100 
#> Test loss: 0.368983 
#> = Starting epoch 101 
#> Test loss: 0.3741205 
#> = Starting epoch 102 
#> Test loss: 0.3769834 
#> = Starting epoch 103 
#> Test loss: 0.3725211 
#> = Starting epoch 104 
#> Test loss: 0.3730062 
#> = Starting epoch 105 
#> Test loss: 0.3728346 
#> = Starting epoch 106 
#> Test loss: 0.3688296 
#> = Starting epoch 107 
#> Test loss: 0.3795431 
#> = Starting epoch 108 
#> Test loss: 0.3816218 
#> = Starting epoch 109 
#> Test loss: 0.3714893 
#> = Starting epoch 110 
#> Test loss: 0.3764892 
#> = Starting epoch 111 
#> Test loss: 0.3775806 
#> = Starting epoch 112 
#> Test loss: 0.3759674 
#> = Starting epoch 113 
#> Test loss: 0.3822531 
#> = Starting epoch 114 
#> Test loss: 0.368648 
#> = Starting epoch 115 
#> Test loss: 0.371012 
#> = Starting epoch 116 
#> Test loss: 0.3715886 
#> = Starting epoch 117 
#> Test loss: 0.3701208 
#> = Starting epoch 118 
#> Test loss: 0.3776811 
#> = Starting epoch 119 
#> Test loss: 0.3679525 
#> = Starting epoch 120 
#> Test loss: 0.3749079 
#> = Starting epoch 121 
#> Test loss: 0.3740299 
#> = Starting epoch 122 
#> Test loss: 0.4205672 
#> = Starting epoch 123 
#> Test loss: 0.4963447 
#> = Starting epoch 124 
#> Test loss: 0.4614847 
#> = Starting epoch 125 
#> Test loss: 0.4029231 
#> = Starting epoch 126 
#> Test loss: 0.3855864 
#> = Starting epoch 127 
#> Test loss: 0.4243133 
#> = Starting epoch 128 
#> Test loss: 0.4253325
```

It looks as though the test loss has converged. We can sample from the
trained flow as follows and compare the samples to the test set:

``` r

test_samples <- as_array(generate_from_conditional_flow(flow_model, 1024))
test_target <- as_array(test_set$target)
plot(test_target, xlab = 'mu', ylab = 'log(sigma)')
points(test_samples, col = 'red')
```

![](torchflow_files/figure-html/unnamed-chunk-13-1.png)

This looks like a reasonable approximation to the target distribution.
We can also look at marginal histograms:

``` r

par(mfrow = c(2, 2))
hist(test_target[, 1], main = 'Target', xlab = 'mu', freq = FALSE, breaks = 32)
hist(test_samples[, 1], main = 'Samples', xlab = 'mu', freq = FALSE, breaks = 32)
hist(test_target[, 2], main = 'Target', xlab = 'log(sigma)', freq = FALSE, breaks = 32)
hist(test_samples[, 2], main = 'Samples', xlab = 'log(sigma)', freq = FALSE, breaks = 32)
```

![](torchflow_files/figure-html/unnamed-chunk-14-1.png)

These look okay.

## Training a conditional flow

The process for training a conditional flow is the same as for an
unconditional flow, except that the `generate` function now also returns
the conditioning variable. Let’s add a conditioning variable,
$`y \sim N(\mu, \sigma^2)`$, with four replicates:

``` r

generate_conditional_samples <- function(...) {
  n_samples <- 1024
  sigma <- torch_abs(torch_randn(n_samples))
  mu <- torch_randn(n_samples) * sigma
  y <- torch_unsqueeze(mu, 2) + torch_randn(n_samples, 4) * torch_unsqueeze(sigma, 2)
  list(
    target = torch_stack(list(mu, torch_log(sigma)), 2),
    conditioning = y
  )
}

# Make a conditional flow
flow_model <- nn_sequential_conditional_flow(
  nn_affine_coupling_block(2, 4),
  nn_permutation_flow(2),
  nn_affine_coupling_block(2, 4),
  nn_permutation_flow(2),
  nn_affine_coupling_block(2, 4)
)

# Generate a test set
test_set <- generate_conditional_samples()
str(test_set)
#> List of 2
#>  $ target      :Float [1:1024, 1:2]
#>  $ conditioning:Float [1:1024, 1:4]

# Train the flow
train_conditional_flow(
  flow_model,
  generate_conditional_samples,
  n_epochs = 256,
  batch_size = 1024,
  after_epoch = function(...) {
    test_loss <- as_array(forward_kl_loss(flow_model(test_set$target, test_set$conditioning)))
    cat('Test loss:', test_loss, '\n')
  }
)
#> = Starting epoch 1 
#> Test loss: 1.235104 
#> = Starting epoch 2 
#> Test loss: 1.161637 
#> = Starting epoch 3 
#> Test loss: 1.087455 
#> = Starting epoch 4 
#> Test loss: 1.013524 
#> = Starting epoch 5 
#> Test loss: 0.9389997 
#> = Starting epoch 6 
#> Test loss: 0.8629651 
#> = Starting epoch 7 
#> Test loss: 0.7831652 
#> = Starting epoch 8 
#> Test loss: 0.6991326 
#> = Starting epoch 9 
#> Test loss: 0.6099796 
#> = Starting epoch 10 
#> Test loss: 0.5157736 
#> = Starting epoch 11 
#> Test loss: 0.4180205 
#> = Starting epoch 12 
#> Test loss: 0.3202264 
#> = Starting epoch 13 
#> Test loss: 0.2246976 
#> = Starting epoch 14 
#> Test loss: 0.1284185 
#> = Starting epoch 15 
#> Test loss: 0.04440379 
#> = Starting epoch 16 
#> Test loss: 0.000856936 
#> = Starting epoch 17 
#> Test loss: -0.021191 
#> = Starting epoch 18 
#> Test loss: -0.03826737 
#> = Starting epoch 19 
#> Test loss: -0.05089629 
#> = Starting epoch 20 
#> Test loss: -0.09157854 
#> = Starting epoch 21 
#> Test loss: -0.1478677 
#> = Starting epoch 22 
#> Test loss: -0.2023 
#> = Starting epoch 23 
#> Test loss: -0.2727855 
#> = Starting epoch 24 
#> Test loss: -0.3354691 
#> = Starting epoch 25 
#> Test loss: -0.381951 
#> = Starting epoch 26 
#> Test loss: -0.4473241 
#> = Starting epoch 27 
#> Test loss: -0.4744091 
#> = Starting epoch 28 
#> Test loss: -0.4999082 
#> = Starting epoch 29 
#> Test loss: -0.5344474 
#> = Starting epoch 30 
#> Test loss: -0.5570746 
#> = Starting epoch 31 
#> Test loss: -0.5872532 
#> = Starting epoch 32 
#> Test loss: -0.6077206 
#> = Starting epoch 33 
#> Test loss: -0.615349 
#> = Starting epoch 34 
#> Test loss: -0.6342636 
#> = Starting epoch 35 
#> Test loss: -0.6656718 
#> = Starting epoch 36 
#> Test loss: -0.6909893 
#> = Starting epoch 37 
#> Test loss: -0.7211207 
#> = Starting epoch 38 
#> Test loss: -0.7488486 
#> = Starting epoch 39 
#> Test loss: -0.7581905 
#> = Starting epoch 40 
#> Test loss: -0.7856451 
#> = Starting epoch 41 
#> Test loss: -0.8341182 
#> = Starting epoch 42 
#> Test loss: -0.8521712 
#> = Starting epoch 43 
#> Test loss: -0.8576968 
#> = Starting epoch 44 
#> Test loss: -0.8896368 
#> = Starting epoch 45 
#> Test loss: -0.9293932 
#> = Starting epoch 46 
#> Test loss: -0.958735 
#> = Starting epoch 47 
#> Test loss: -0.9539469 
#> = Starting epoch 48 
#> Test loss: -0.9658142 
#> = Starting epoch 49 
#> Test loss: -0.9697835 
#> = Starting epoch 50 
#> Test loss: -1.012259 
#> = Starting epoch 51 
#> Test loss: -1.043931 
#> = Starting epoch 52 
#> Test loss: -1.064102 
#> = Starting epoch 53 
#> Test loss: -1.071424 
#> = Starting epoch 54 
#> Test loss: -1.075763 
#> = Starting epoch 55 
#> Test loss: -1.080834 
#> = Starting epoch 56 
#> Test loss: -1.092009 
#> = Starting epoch 57 
#> Test loss: -1.1076 
#> = Starting epoch 58 
#> Test loss: -1.126339 
#> = Starting epoch 59 
#> Test loss: -1.088287 
#> = Starting epoch 60 
#> Test loss: -1.163872 
#> = Starting epoch 61 
#> Test loss: -1.120451 
#> = Starting epoch 62 
#> Test loss: -1.1733 
#> = Starting epoch 63 
#> Test loss: -1.164774 
#> = Starting epoch 64 
#> Test loss: -1.189823 
#> = Starting epoch 65 
#> Test loss: -1.163503 
#> = Starting epoch 66 
#> Test loss: -1.194759 
#> = Starting epoch 67 
#> Test loss: -1.1436 
#> = Starting epoch 68 
#> Test loss: -1.201026 
#> = Starting epoch 69 
#> Test loss: -1.179689 
#> = Starting epoch 70 
#> Test loss: -1.213942 
#> = Starting epoch 71 
#> Test loss: -1.191445 
#> = Starting epoch 72 
#> Test loss: -1.208952 
#> = Starting epoch 73 
#> Test loss: -1.153196 
#> = Starting epoch 74 
#> Test loss: -1.245654 
#> = Starting epoch 75 
#> Test loss: -1.164784 
#> = Starting epoch 76 
#> Test loss: -1.251997 
#> = Starting epoch 77 
#> Test loss: -1.194567 
#> = Starting epoch 78 
#> Test loss: -1.196761 
#> = Starting epoch 79 
#> Test loss: -1.258924 
#> = Starting epoch 80 
#> Test loss: -1.219843 
#> = Starting epoch 81 
#> Test loss: -1.141532 
#> = Starting epoch 82 
#> Test loss: -1.123561 
#> = Starting epoch 83 
#> Test loss: -1.171093 
#> = Starting epoch 84 
#> Test loss: -1.223439 
#> = Starting epoch 85 
#> Test loss: -1.177964 
#> = Starting epoch 86 
#> Test loss: -1.093544 
#> = Starting epoch 87 
#> Test loss: -1.123199 
#> = Starting epoch 88 
#> Test loss: -1.214433 
#> = Starting epoch 89 
#> Test loss: -1.224938 
#> = Starting epoch 90 
#> Test loss: -1.16787 
#> = Starting epoch 91 
#> Test loss: -1.191574 
#> = Starting epoch 92 
#> Test loss: -1.252049 
#> = Starting epoch 93 
#> Test loss: -1.264398 
#> = Starting epoch 94 
#> Test loss: -1.229466 
#> = Starting epoch 95 
#> Test loss: -1.22453 
#> = Starting epoch 96 
#> Test loss: -1.263436 
#> = Starting epoch 97 
#> Test loss: -1.275043 
#> = Starting epoch 98 
#> Test loss: -1.236719 
#> = Starting epoch 99 
#> Test loss: -1.210205 
#> = Starting epoch 100 
#> Test loss: -1.204618 
#> = Starting epoch 101 
#> Test loss: -1.235753 
#> = Starting epoch 102 
#> Test loss: -1.282313 
#> = Starting epoch 103 
#> Test loss: -1.292845 
#> = Starting epoch 104 
#> Test loss: -1.271426 
#> = Starting epoch 105 
#> Test loss: -1.270918 
#> = Starting epoch 106 
#> Test loss: -1.288683 
#> = Starting epoch 107 
#> Test loss: -1.309901 
#> = Starting epoch 108 
#> Test loss: -1.299946 
#> = Starting epoch 109 
#> Test loss: -1.313279 
#> = Starting epoch 110 
#> Test loss: -1.305623 
#> = Starting epoch 111 
#> Test loss: -1.293514 
#> = Starting epoch 112 
#> Test loss: -1.297937 
#> = Starting epoch 113 
#> Test loss: -1.314655 
#> = Starting epoch 114 
#> Test loss: -1.32425 
#> = Starting epoch 115 
#> Test loss: -1.319244 
#> = Starting epoch 116 
#> Test loss: -1.309697 
#> = Starting epoch 117 
#> Test loss: -1.319812 
#> = Starting epoch 118 
#> Test loss: -1.32695 
#> = Starting epoch 119 
#> Test loss: -1.316109 
#> = Starting epoch 120 
#> Test loss: -1.306357 
#> = Starting epoch 121 
#> Test loss: -1.312207 
#> = Starting epoch 122 
#> Test loss: -1.324086 
#> = Starting epoch 123 
#> Test loss: -1.318377 
#> = Starting epoch 124 
#> Test loss: -1.307313 
#> = Starting epoch 125 
#> Test loss: -1.312105 
#> = Starting epoch 126 
#> Test loss: -1.334633 
#> = Starting epoch 127 
#> Test loss: -1.342663 
#> = Starting epoch 128 
#> Test loss: -1.314122 
#> = Starting epoch 129 
#> Test loss: -1.303788 
#> = Starting epoch 130 
#> Test loss: -1.329466 
#> = Starting epoch 131 
#> Test loss: -1.334765 
#> = Starting epoch 132 
#> Test loss: -1.310944 
#> = Starting epoch 133 
#> Test loss: -1.313559 
#> = Starting epoch 134 
#> Test loss: -1.341197 
#> = Starting epoch 135 
#> Test loss: -1.340664 
#> = Starting epoch 136 
#> Test loss: -1.321438 
#> = Starting epoch 137 
#> Test loss: -1.321307 
#> = Starting epoch 138 
#> Test loss: -1.339173 
#> = Starting epoch 139 
#> Test loss: -1.353055 
#> = Starting epoch 140 
#> Test loss: -1.35797 
#> = Starting epoch 141 
#> Test loss: -1.343075 
#> = Starting epoch 142 
#> Test loss: -1.330583 
#> = Starting epoch 143 
#> Test loss: -1.343359 
#> = Starting epoch 144 
#> Test loss: -1.340493 
#> = Starting epoch 145 
#> Test loss: -1.34442 
#> = Starting epoch 146 
#> Test loss: -1.352065 
#> = Starting epoch 147 
#> Test loss: -1.361777 
#> = Starting epoch 148 
#> Test loss: -1.354138 
#> = Starting epoch 149 
#> Test loss: -1.352241 
#> = Starting epoch 150 
#> Test loss: -1.359106 
#> = Starting epoch 151 
#> Test loss: -1.363906 
#> = Starting epoch 152 
#> Test loss: -1.364966 
#> = Starting epoch 153 
#> Test loss: -1.364846 
#> = Starting epoch 154 
#> Test loss: -1.374705 
#> = Starting epoch 155 
#> Test loss: -1.367807 
#> = Starting epoch 156 
#> Test loss: -1.365253 
#> = Starting epoch 157 
#> Test loss: -1.376521 
#> = Starting epoch 158 
#> Test loss: -1.372672 
#> = Starting epoch 159 
#> Test loss: -1.372299 
#> = Starting epoch 160 
#> Test loss: -1.364572 
#> = Starting epoch 161 
#> Test loss: -1.361834 
#> = Starting epoch 162 
#> Test loss: -1.370253 
#> = Starting epoch 163 
#> Test loss: -1.369457 
#> = Starting epoch 164 
#> Test loss: -1.360791 
#> = Starting epoch 165 
#> Test loss: -1.375546 
#> = Starting epoch 166 
#> Test loss: -1.378762 
#> = Starting epoch 167 
#> Test loss: -1.372631 
#> = Starting epoch 168 
#> Test loss: -1.375416 
#> = Starting epoch 169 
#> Test loss: -1.38777 
#> = Starting epoch 170 
#> Test loss: -1.385056 
#> = Starting epoch 171 
#> Test loss: -1.379745 
#> = Starting epoch 172 
#> Test loss: -1.384876 
#> = Starting epoch 173 
#> Test loss: -1.387825 
#> = Starting epoch 174 
#> Test loss: -1.383155 
#> = Starting epoch 175 
#> Test loss: -1.389258 
#> = Starting epoch 176 
#> Test loss: -1.394925 
#> = Starting epoch 177 
#> Test loss: -1.388434 
#> = Starting epoch 178 
#> Test loss: -1.382191 
#> = Starting epoch 179 
#> Test loss: -1.383497 
#> = Starting epoch 180 
#> Test loss: -1.38093 
#> = Starting epoch 181 
#> Test loss: -1.386164 
#> = Starting epoch 182 
#> Test loss: -1.395264 
#> = Starting epoch 183 
#> Test loss: -1.397324 
#> = Starting epoch 184 
#> Test loss: -1.392619 
#> = Starting epoch 185 
#> Test loss: -1.396527 
#> = Starting epoch 186 
#> Test loss: -1.399156 
#> = Starting epoch 187 
#> Test loss: -1.395388 
#> = Starting epoch 188 
#> Test loss: -1.390759 
#> = Starting epoch 189 
#> Test loss: -1.374961 
#> = Starting epoch 190 
#> Test loss: -1.381762 
#> = Starting epoch 191 
#> Test loss: -1.393375 
#> = Starting epoch 192 
#> Test loss: -1.400173 
#> = Starting epoch 193 
#> Test loss: -1.400796 
#> = Starting epoch 194 
#> Test loss: -1.397059 
#> = Starting epoch 195 
#> Test loss: -1.398163 
#> = Starting epoch 196 
#> Test loss: -1.402197 
#> = Starting epoch 197 
#> Test loss: -1.408081 
#> = Starting epoch 198 
#> Test loss: -1.404255 
#> = Starting epoch 199 
#> Test loss: -1.400918 
#> = Starting epoch 200 
#> Test loss: -1.389056 
#> = Starting epoch 201 
#> Test loss: -1.37326 
#> = Starting epoch 202 
#> Test loss: -1.387016 
#> = Starting epoch 203 
#> Test loss: -1.390213 
#> = Starting epoch 204 
#> Test loss: -1.391067 
#> = Starting epoch 205 
#> Test loss: -1.399378 
#> = Starting epoch 206 
#> Test loss: -1.400293 
#> = Starting epoch 207 
#> Test loss: -1.401732 
#> = Starting epoch 208 
#> Test loss: -1.404093 
#> = Starting epoch 209 
#> Test loss: -1.408671 
#> = Starting epoch 210 
#> Test loss: -1.406441 
#> = Starting epoch 211 
#> Test loss: -1.400244 
#> = Starting epoch 212 
#> Test loss: -1.395388 
#> = Starting epoch 213 
#> Test loss: -1.39179 
#> = Starting epoch 214 
#> Test loss: -1.402482 
#> = Starting epoch 215 
#> Test loss: -1.409302 
#> = Starting epoch 216 
#> Test loss: -1.407092 
#> = Starting epoch 217 
#> Test loss: -1.407213 
#> = Starting epoch 218 
#> Test loss: -1.392236 
#> = Starting epoch 219 
#> Test loss: -1.398014 
#> = Starting epoch 220 
#> Test loss: -1.389201 
#> = Starting epoch 221 
#> Test loss: -1.403805 
#> = Starting epoch 222 
#> Test loss: -1.379252 
#> = Starting epoch 223 
#> Test loss: -1.391586 
#> = Starting epoch 224 
#> Test loss: -1.298837 
#> = Starting epoch 225 
#> Test loss: -1.333828 
#> = Starting epoch 226 
#> Test loss: -1.404978 
#> = Starting epoch 227 
#> Test loss: -1.379434 
#> = Starting epoch 228 
#> Test loss: -1.359704 
#> = Starting epoch 229 
#> Test loss: -1.412236 
#> = Starting epoch 230 
#> Test loss: -1.391483 
#> = Starting epoch 231 
#> Test loss: -1.364606 
#> = Starting epoch 232 
#> Test loss: -1.372846 
#> = Starting epoch 233 
#> Test loss: -1.403506 
#> = Starting epoch 234 
#> Test loss: -1.385739 
#> = Starting epoch 235 
#> Test loss: -1.373251 
#> = Starting epoch 236 
#> Test loss: -1.389808 
#> = Starting epoch 237 
#> Test loss: -1.384222 
#> = Starting epoch 238 
#> Test loss: -1.379506 
#> = Starting epoch 239 
#> Test loss: -1.401834 
#> = Starting epoch 240 
#> Test loss: -1.401507 
#> = Starting epoch 241 
#> Test loss: -1.401339 
#> = Starting epoch 242 
#> Test loss: -1.39976 
#> = Starting epoch 243 
#> Test loss: -1.404801 
#> = Starting epoch 244 
#> Test loss: -1.399129 
#> = Starting epoch 245 
#> Test loss: -1.410925 
#> = Starting epoch 246 
#> Test loss: -1.408674 
#> = Starting epoch 247 
#> Test loss: -1.40327 
#> = Starting epoch 248 
#> Test loss: -1.410019 
#> = Starting epoch 249 
#> Test loss: -1.410446 
#> = Starting epoch 250 
#> Test loss: -1.402437 
#> = Starting epoch 251 
#> Test loss: -1.423491 
#> = Starting epoch 252 
#> Test loss: -1.400958 
#> = Starting epoch 253 
#> Test loss: -1.410783 
#> = Starting epoch 254 
#> Test loss: -1.419274 
#> = Starting epoch 255 
#> Test loss: -1.410238 
#> = Starting epoch 256 
#> Test loss: -1.413607
```

We can generate samples from the trained flow as follows, where now the
samples are conditioned on the values of $`y`$:

``` r

# Generate 1024 samples for each of the first 4 conditioning variables in the test set
test_samples <- as_array(generate_from_conditional_flow(flow_model, 1024, test_set$conditioning[1 : 4, ]))
test_target <- as_array(test_set$target)

par(mfrow = c(4, 2))
for (i in 1 : 4) {
  hist(test_samples[, i, 1], main = '', xlab = 'mu', freq = FALSE, breaks = 32, xlim = c(-3, 3))
  abline(v = as_array(test_set$conditioning[i, ]), col = 'blue')
  abline(v = test_target[i, 1], col = 'red')
  hist(test_samples[, i, 2], main = '', xlab = 'log(sigma)', freq = FALSE, breaks = 32, xlim = c(-8, 8))
  abline(v = test_target[i, 2], col = 'red')
}
```

![](torchflow_files/figure-html/unnamed-chunk-16-1.png)

We can also plot the samples on a scatter plot:

``` r

par(mfrow = c(2, 2))
for (i in 1 : 4) {
  plot(
    test_samples[, i, 1], test_samples[, i, 2], main = '',
    xlab = 'mu', ylab = 'log(sigma)', xlim = c(-3, 3), ylim = c(-8, 8)
  )
  abline(v = test_target[i, 1], col = 'red')
  abline(h = test_target[i, 2], col = 'red')
}
```

![](torchflow_files/figure-html/unnamed-chunk-17-1.png)

**Compare these to MCMC**

## Using a summarizing network

In the above example, the conditioning variable `y` contains four
replicates of the conditioning variable. We ignored the fact that these
are replicates, and the trained the flow as though they could be
dependent. We can instead use a summarizing network that individually
processes each individual replicate into a set of summary statistics,
and then combine the summary statistics in a permutation invariant way
to form the conditioning variable. Here is an example summarizing
network:

``` r

# Helper modules
nn_sum <- nn_module(
  initialize = function(dimension) {
    self$dimension <- dimension
  },
  forward = function(x) {
    torch_sum(x, self$dimension)
  }
)

nn_unsqueeze <- nn_module(
  initialize = function(dimension) {
    self$dimension <- dimension
  },
  forward = function(x) {
    torch_unsqueeze(x, self$dimension)
  }
)

summary_model <- nn_sequential(
  # Add an extra unit dimension for the replicate dimension
  nn_unsqueeze(-1),
  # Compute the summary statistics for each replicate
  nn_linear(1, 32),
  nn_relu(),
  nn_linear(32, 32),
  nn_relu(),
  nn_linear(32, 8),
  # Sum the summary statistics across the replicates
  nn_sum(-2)
)

summary_model(test_set$conditioning[1 : 10, , drop = FALSE])
#> torch_tensor
#>  0.8808 -0.4516  0.1276  0.6069 -0.8523 -0.8365 -0.8911 -0.0145
#>  1.0112 -0.5399 -0.1001  0.6504 -1.0432 -0.8209 -1.1022  0.1204
#>  0.9269 -0.4934  0.2573  0.5437 -0.7589 -0.8374 -0.7930 -0.0201
#>  0.9506 -0.6324  0.3409  0.4918 -0.6635 -0.7574 -0.6561  0.0631
#>  0.9478 -0.5384  0.3100  0.5208 -0.7103 -0.8228 -0.7350 -0.0048
#>  0.9552 -0.4934  0.2871  0.5203 -0.7514 -0.8440 -0.8086 -0.0396
#>  2.2429 -1.0575 -1.3244  0.5820 -1.5739 -1.0249 -2.5322  0.8417
#>  0.9183 -0.4747  0.1104  0.6099 -0.8746 -0.8269 -0.9093  0.0082
#>  0.9431 -0.6318  0.1211  0.5878 -0.8310 -0.7464 -0.8295  0.1457
#>  0.9778 -0.7154  0.3942  0.4612 -0.5955 -0.7155 -0.5599  0.1035
#> [ CPUFloatType{10,8} ][ grad_fn = <SumBackward1> ]
```

We can combine the summarizing network with the flow in a
`nn_summarizing_conditional_flow` object:

``` r

flow_model <- nn_sequential_conditional_flow(
  nn_affine_coupling_block(2, 8),
  nn_permutation_flow(2),
  nn_affine_coupling_block(2, 8),
  nn_permutation_flow(2),
  nn_affine_coupling_block(2, 8)
)
summarizing_flow_model <- nn_summarizing_conditional_flow(summary_model, flow_model)
```

Let’s also expand the number of replicated observations to 32:

``` r

generate_conditional_samples <- function(...) {
  n_samples <- 1024
  sigma <- torch_abs(torch_randn(n_samples))
  mu <- torch_randn(n_samples) * sigma
  y <- torch_unsqueeze(mu, 2) + torch_randn(n_samples, 32) * torch_unsqueeze(sigma, 2)
  list(
    target = torch_stack(list(mu, torch_log(sigma)), 2),
    conditioning = y
  )
}
```

Let’s train the model:

``` r

test_set <- generate_conditional_samples()
str(test_set)
#> List of 2
#>  $ target      :Float [1:1024, 1:2]
#>  $ conditioning:Float [1:1024, 1:32]
train_conditional_flow(
  summarizing_flow_model,
  generate_conditional_samples,
  n_epochs = 256,
  batch_size = 1024,
  after_epoch = function(...) {
    test_loss <- as_array(forward_kl_loss(summarizing_flow_model(test_set$target, test_set$conditioning)))
    cat('Test loss:', test_loss, '\n')
  }
)
#> = Starting epoch 1 
#> Test loss: 1.056542 
#> = Starting epoch 2 
#> Test loss: 0.9386197 
#> = Starting epoch 3 
#> Test loss: 0.8395875 
#> = Starting epoch 4 
#> Test loss: 0.6740577 
#> = Starting epoch 5 
#> Test loss: 0.4795395 
#> = Starting epoch 6 
#> Test loss: 0.2887294 
#> = Starting epoch 7 
#> Test loss: 0.1006605 
#> = Starting epoch 8 
#> Test loss: -0.1061829 
#> = Starting epoch 9 
#> Test loss: -0.348986 
#> = Starting epoch 10 
#> Test loss: -0.5937135 
#> = Starting epoch 11 
#> Test loss: -0.8172616 
#> = Starting epoch 12 
#> Test loss: -1.033609 
#> = Starting epoch 13 
#> Test loss: -1.169163 
#> = Starting epoch 14 
#> Test loss: -1.276567 
#> = Starting epoch 15 
#> Test loss: -1.120428 
#> = Starting epoch 16 
#> Test loss: -1.445064 
#> = Starting epoch 17 
#> Test loss: -1.372398 
#> = Starting epoch 18 
#> Test loss: -1.557822 
#> = Starting epoch 19 
#> Test loss: -1.585135 
#> = Starting epoch 20 
#> Test loss: -1.590711 
#> = Starting epoch 21 
#> Test loss: -1.705605 
#> = Starting epoch 22 
#> Test loss: -1.741658 
#> = Starting epoch 23 
#> Test loss: -1.816008 
#> = Starting epoch 24 
#> Test loss: -1.856513 
#> = Starting epoch 25 
#> Test loss: -1.876365 
#> = Starting epoch 26 
#> Test loss: -1.882213 
#> = Starting epoch 27 
#> Test loss: -1.972118 
#> = Starting epoch 28 
#> Test loss: -2.068038 
#> = Starting epoch 29 
#> Test loss: -2.09432 
#> = Starting epoch 30 
#> Test loss: -2.135259 
#> = Starting epoch 31 
#> Test loss: -2.114792 
#> = Starting epoch 32 
#> Test loss: -2.205391 
#> = Starting epoch 33 
#> Test loss: -2.186504 
#> = Starting epoch 34 
#> Test loss: -2.248096 
#> = Starting epoch 35 
#> Test loss: -2.278724 
#> = Starting epoch 36 
#> Test loss: -2.311905 
#> = Starting epoch 37 
#> Test loss: -2.380715 
#> = Starting epoch 38 
#> Test loss: -2.363258 
#> = Starting epoch 39 
#> Test loss: -2.420071 
#> = Starting epoch 40 
#> Test loss: -2.478081 
#> = Starting epoch 41 
#> Test loss: -2.483109 
#> = Starting epoch 42 
#> Test loss: -2.507818 
#> = Starting epoch 43 
#> Test loss: -2.549404 
#> = Starting epoch 44 
#> Test loss: -2.589152 
#> = Starting epoch 45 
#> Test loss: -2.574725 
#> = Starting epoch 46 
#> Test loss: -2.596677 
#> = Starting epoch 47 
#> Test loss: -2.548864 
#> = Starting epoch 48 
#> Test loss: -2.530041 
#> = Starting epoch 49 
#> Test loss: -2.626273 
#> = Starting epoch 50 
#> Test loss: -2.615583 
#> = Starting epoch 51 
#> Test loss: -2.50384 
#> = Starting epoch 52 
#> Test loss: -2.712681 
#> = Starting epoch 53 
#> Test loss: -2.45332 
#> = Starting epoch 54 
#> Test loss: -2.651956 
#> = Starting epoch 55 
#> Test loss: -2.666153 
#> = Starting epoch 56 
#> Test loss: -2.6177 
#> = Starting epoch 57 
#> Test loss: -2.6903 
#> = Starting epoch 58 
#> Test loss: -2.685179 
#> = Starting epoch 59 
#> Test loss: -2.705271 
#> = Starting epoch 60 
#> Test loss: -2.736803 
#> = Starting epoch 61 
#> Test loss: -2.736627 
#> = Starting epoch 62 
#> Test loss: -2.784014 
#> = Starting epoch 63 
#> Test loss: -2.734004 
#> = Starting epoch 64 
#> Test loss: -2.729198 
#> = Starting epoch 65 
#> Test loss: -2.779279 
#> = Starting epoch 66 
#> Test loss: -2.808273 
#> = Starting epoch 67 
#> Test loss: -2.834967 
#> = Starting epoch 68 
#> Test loss: -2.80118 
#> = Starting epoch 69 
#> Test loss: -2.850321 
#> = Starting epoch 70 
#> Test loss: -2.812217 
#> = Starting epoch 71 
#> Test loss: -2.860163 
#> = Starting epoch 72 
#> Test loss: -2.858403 
#> = Starting epoch 73 
#> Test loss: -2.868072 
#> = Starting epoch 74 
#> Test loss: -2.86394 
#> = Starting epoch 75 
#> Test loss: -2.873101 
#> = Starting epoch 76 
#> Test loss: -2.866317 
#> = Starting epoch 77 
#> Test loss: -2.82303 
#> = Starting epoch 78 
#> Test loss: -2.874326 
#> = Starting epoch 79 
#> Test loss: -2.91161 
#> = Starting epoch 80 
#> Test loss: -2.894839 
#> = Starting epoch 81 
#> Test loss: -2.821064 
#> = Starting epoch 82 
#> Test loss: -2.797119 
#> = Starting epoch 83 
#> Test loss: -2.880533 
#> = Starting epoch 84 
#> Test loss: -2.856995 
#> = Starting epoch 85 
#> Test loss: -2.647371 
#> = Starting epoch 86 
#> Test loss: -2.927388 
#> = Starting epoch 87 
#> Test loss: -2.898995 
#> = Starting epoch 88 
#> Test loss: -2.899976 
#> = Starting epoch 89 
#> Test loss: -2.952538 
#> = Starting epoch 90 
#> Test loss: -2.884227 
#> = Starting epoch 91 
#> Test loss: -2.943613 
#> = Starting epoch 92 
#> Test loss: -2.843575 
#> = Starting epoch 93 
#> Test loss: -2.869985 
#> = Starting epoch 94 
#> Test loss: -2.971624 
#> = Starting epoch 95 
#> Test loss: -2.797068 
#> = Starting epoch 96 
#> Test loss: -2.973121 
#> = Starting epoch 97 
#> Test loss: -2.913483 
#> = Starting epoch 98 
#> Test loss: -2.930388 
#> = Starting epoch 99 
#> Test loss: -2.986427 
#> = Starting epoch 100 
#> Test loss: -2.931566 
#> = Starting epoch 101 
#> Test loss: -3.00165 
#> = Starting epoch 102 
#> Test loss: -2.948899 
#> = Starting epoch 103 
#> Test loss: -3.002074 
#> = Starting epoch 104 
#> Test loss: -3.021454 
#> = Starting epoch 105 
#> Test loss: -2.976741 
#> = Starting epoch 106 
#> Test loss: -2.99652 
#> = Starting epoch 107 
#> Test loss: -3.010398 
#> = Starting epoch 108 
#> Test loss: -2.974086 
#> = Starting epoch 109 
#> Test loss: -3.018012 
#> = Starting epoch 110 
#> Test loss: -3.011648 
#> = Starting epoch 111 
#> Test loss: -2.98257 
#> = Starting epoch 112 
#> Test loss: -3.028962 
#> = Starting epoch 113 
#> Test loss: -3.036261 
#> = Starting epoch 114 
#> Test loss: -3.010018 
#> = Starting epoch 115 
#> Test loss: -3.054339 
#> = Starting epoch 116 
#> Test loss: -3.032013 
#> = Starting epoch 117 
#> Test loss: -2.995591 
#> = Starting epoch 118 
#> Test loss: -3.033182 
#> = Starting epoch 119 
#> Test loss: -3.055732 
#> = Starting epoch 120 
#> Test loss: -3.029338 
#> = Starting epoch 121 
#> Test loss: -3.051992 
#> = Starting epoch 122 
#> Test loss: -3.02594 
#> = Starting epoch 123 
#> Test loss: -3.036039 
#> = Starting epoch 124 
#> Test loss: -3.040451 
#> = Starting epoch 125 
#> Test loss: -3.037863 
#> = Starting epoch 126 
#> Test loss: -3.012582 
#> = Starting epoch 127 
#> Test loss: -2.987597 
#> = Starting epoch 128 
#> Test loss: -3.052055 
#> = Starting epoch 129 
#> Test loss: -3.063831 
#> = Starting epoch 130 
#> Test loss: -3.065559 
#> = Starting epoch 131 
#> Test loss: -3.084764 
#> = Starting epoch 132 
#> Test loss: -3.069906 
#> = Starting epoch 133 
#> Test loss: -3.079223 
#> = Starting epoch 134 
#> Test loss: -3.049434 
#> = Starting epoch 135 
#> Test loss: -3.071605 
#> = Starting epoch 136 
#> Test loss: -3.073457 
#> = Starting epoch 137 
#> Test loss: -3.046532 
#> = Starting epoch 138 
#> Test loss: -3.033834 
#> = Starting epoch 139 
#> Test loss: -3.030584 
#> = Starting epoch 140 
#> Test loss: -3.032061 
#> = Starting epoch 141 
#> Test loss: -3.078155 
#> = Starting epoch 142 
#> Test loss: -3.090306 
#> = Starting epoch 143 
#> Test loss: -3.084477 
#> = Starting epoch 144 
#> Test loss: -3.061762 
#> = Starting epoch 145 
#> Test loss: -3.061894 
#> = Starting epoch 146 
#> Test loss: -3.031111 
#> = Starting epoch 147 
#> Test loss: -3.060192 
#> = Starting epoch 148 
#> Test loss: -3.047997 
#> = Starting epoch 149 
#> Test loss: -3.042882 
#> = Starting epoch 150 
#> Test loss: -3.109193 
#> = Starting epoch 151 
#> Test loss: -3.055078 
#> = Starting epoch 152 
#> Test loss: -3.100046 
#> = Starting epoch 153 
#> Test loss: -3.126127 
#> = Starting epoch 154 
#> Test loss: -3.119638 
#> = Starting epoch 155 
#> Test loss: -3.084233 
#> = Starting epoch 156 
#> Test loss: -3.06316 
#> = Starting epoch 157 
#> Test loss: -2.997062 
#> = Starting epoch 158 
#> Test loss: -2.998207 
#> = Starting epoch 159 
#> Test loss: -3.060229 
#> = Starting epoch 160 
#> Test loss: -3.112577 
#> = Starting epoch 161 
#> Test loss: -3.091738 
#> = Starting epoch 162 
#> Test loss: -3.099608 
#> = Starting epoch 163 
#> Test loss: -3.058151 
#> = Starting epoch 164 
#> Test loss: -3.098911 
#> = Starting epoch 165 
#> Test loss: -3.109108 
#> = Starting epoch 166 
#> Test loss: -3.057822 
#> = Starting epoch 167 
#> Test loss: -3.089241 
#> = Starting epoch 168 
#> Test loss: -3.013163 
#> = Starting epoch 169 
#> Test loss: -3.110768 
#> = Starting epoch 170 
#> Test loss: -3.101031 
#> = Starting epoch 171 
#> Test loss: -3.058825 
#> = Starting epoch 172 
#> Test loss: -3.068336 
#> = Starting epoch 173 
#> Test loss: -3.114203 
#> = Starting epoch 174 
#> Test loss: -3.114196 
#> = Starting epoch 175 
#> Test loss: -3.121705 
#> = Starting epoch 176 
#> Test loss: -3.046431 
#> = Starting epoch 177 
#> Test loss: -3.081213 
#> = Starting epoch 178 
#> Test loss: -3.135944 
#> = Starting epoch 179 
#> Test loss: -3.084801 
#> = Starting epoch 180 
#> Test loss: -3.044056 
#> = Starting epoch 181 
#> Test loss: -2.929623 
#> = Starting epoch 182 
#> Test loss: -2.972875 
#> = Starting epoch 183 
#> Test loss: -3.107115 
#> = Starting epoch 184 
#> Test loss: -2.95728 
#> = Starting epoch 185 
#> Test loss: -2.810732 
#> = Starting epoch 186 
#> Test loss: -3.068263 
#> = Starting epoch 187 
#> Test loss: -3.045385 
#> = Starting epoch 188 
#> Test loss: -2.945726 
#> = Starting epoch 189 
#> Test loss: -3.093689 
#> = Starting epoch 190 
#> Test loss: -3.018431 
#> = Starting epoch 191 
#> Test loss: -3.031064 
#> = Starting epoch 192 
#> Test loss: -3.100938 
#> = Starting epoch 193 
#> Test loss: -2.978924 
#> = Starting epoch 194 
#> Test loss: -3.064328 
#> = Starting epoch 195 
#> Test loss: -3.11476 
#> = Starting epoch 196 
#> Test loss: -2.938916 
#> = Starting epoch 197 
#> Test loss: -3.03276 
#> = Starting epoch 198 
#> Test loss: -3.139549 
#> = Starting epoch 199 
#> Test loss: -2.905089 
#> = Starting epoch 200 
#> Test loss: -3.047354 
#> = Starting epoch 201 
#> Test loss: -3.108354 
#> = Starting epoch 202 
#> Test loss: -2.958102 
#> = Starting epoch 203 
#> Test loss: -3.15145 
#> = Starting epoch 204 
#> Test loss: -3.020781 
#> = Starting epoch 205 
#> Test loss: -3.114461 
#> = Starting epoch 206 
#> Test loss: -3.056366 
#> = Starting epoch 207 
#> Test loss: -3.129591 
#> = Starting epoch 208 
#> Test loss: -3.12358 
#> = Starting epoch 209 
#> Test loss: -3.101906 
#> = Starting epoch 210 
#> Test loss: -3.115268 
#> = Starting epoch 211 
#> Test loss: -3.140951 
#> = Starting epoch 212 
#> Test loss: -3.148306 
#> = Starting epoch 213 
#> Test loss: -3.113291 
#> = Starting epoch 214 
#> Test loss: -3.176642 
#> = Starting epoch 215 
#> Test loss: -3.160311 
#> = Starting epoch 216 
#> Test loss: -3.158305 
#> = Starting epoch 217 
#> Test loss: -3.15784 
#> = Starting epoch 218 
#> Test loss: -3.134449 
#> = Starting epoch 219 
#> Test loss: -3.127414 
#> = Starting epoch 220 
#> Test loss: -3.15876 
#> = Starting epoch 221 
#> Test loss: -3.138489 
#> = Starting epoch 222 
#> Test loss: -3.141831 
#> = Starting epoch 223 
#> Test loss: -3.154915 
#> = Starting epoch 224 
#> Test loss: -3.141156 
#> = Starting epoch 225 
#> Test loss: -3.169217 
#> = Starting epoch 226 
#> Test loss: -3.158553 
#> = Starting epoch 227 
#> Test loss: -3.172288 
#> = Starting epoch 228 
#> Test loss: -3.172488 
#> = Starting epoch 229 
#> Test loss: -3.161778 
#> = Starting epoch 230 
#> Test loss: -3.13713 
#> = Starting epoch 231 
#> Test loss: -3.175882 
#> = Starting epoch 232 
#> Test loss: -3.150422 
#> = Starting epoch 233 
#> Test loss: -3.170208 
#> = Starting epoch 234 
#> Test loss: -3.139156 
#> = Starting epoch 235 
#> Test loss: -3.159464 
#> = Starting epoch 236 
#> Test loss: -3.111871 
#> = Starting epoch 237 
#> Test loss: -3.176301 
#> = Starting epoch 238 
#> Test loss: -3.123937 
#> = Starting epoch 239 
#> Test loss: -3.160495 
#> = Starting epoch 240 
#> Test loss: -3.105833 
#> = Starting epoch 241 
#> Test loss: -3.150261 
#> = Starting epoch 242 
#> Test loss: -3.163246 
#> = Starting epoch 243 
#> Test loss: -3.147137 
#> = Starting epoch 244 
#> Test loss: -3.090566 
#> = Starting epoch 245 
#> Test loss: -2.909898 
#> = Starting epoch 246 
#> Test loss: -2.948209 
#> = Starting epoch 247 
#> Test loss: -3.101851 
#> = Starting epoch 248 
#> Test loss: -2.871951 
#> = Starting epoch 249 
#> Test loss: -2.750423 
#> = Starting epoch 250 
#> Test loss: -3.144527 
#> = Starting epoch 251 
#> Test loss: -2.885798 
#> = Starting epoch 252 
#> Test loss: -3.002835 
#> = Starting epoch 253 
#> Test loss: -3.106793 
#> = Starting epoch 254 
#> Test loss: -2.965309 
#> = Starting epoch 255 
#> Test loss: -3.097294 
#> = Starting epoch 256 
#> Test loss: -3.061787
```

We can generate samples from the trained model as before:

``` r

# Generate 1024 samples for each of the first 4 conditioning variables in the test set
test_samples <- as_array(generate_from_conditional_flow(
  summarizing_flow_model,
  1024,
  test_set$conditioning[1 : 4, ]
))
test_target <- as_array(test_set$target)

par(mfrow = c(4, 2))
for (i in 1 : 4) {
  hist(test_samples[, i, 1], main = '', xlab = 'mu', freq = FALSE, breaks = 32, xlim = c(-5, 5))
  abline(v = test_target[i, 1], col = 'red')
  hist(test_samples[, i, 2], main = '', xlab = 'log(sigma)', freq = FALSE, breaks = 32, xlim = c(-3, 3))
  abline(v = test_target[i, 2], col = 'red')
}
```

![](torchflow_files/figure-html/unnamed-chunk-22-1.png)

We can also plot the samples on a scatter plot:

``` r

par(mfrow = c(2, 2))
for (i in 1 : 4) {
  plot(
    test_samples[, i, 1], test_samples[, i, 2], main = '',
    xlab = 'mu', ylab = 'log(sigma)', xlim = c(-5, 5), ylim = c(-3, 3)
  )
  abline(v = test_target[i, 1], col = 'red')
  abline(h = test_target[i, 2], col = 'red')
}
```

![](torchflow_files/figure-html/unnamed-chunk-23-1.png)

## More complex conditioning variables

The summarizing network is the key to using more complex conditioning
variables. For example, we can use a 2D grid of points as the
conditioning variable, which is processed by the summarizing network
into a set of summary statistics which are then used as the conditioning
variable for the flow. For the 2-D grid, a convolutional network is a
conventional choice that often works well in practice.

The following example generates data using an exponential covariance
with unknown variance and length scale over a 16x16 2-D grid:

``` r

n_grid <- 16
x_y_grid <- as.matrix(expand.grid(
  x = seq(0, 1, length.out = n_grid),
  y = seq(0, 1, length.out = n_grid)
))
distances <- torch_tensor(as.matrix(dist(x_y_grid)))

generate_conditional_samples <- function(...) {
  n_samples <- 1024
  ell <- 0.1 + 1.9 * torch_rand(n_samples)
  sigma <- torch_abs(torch_randn(n_samples))

  # Generate the conditioning variable
  Sigma <- (
    torch_square(sigma)$unsqueeze(-1)$unsqueeze(-1) * torch_exp(
      -torch_unsqueeze(distances, 1) / ell$unsqueeze(-1)$unsqueeze(-1)
    )
  )
  L <- linalg_cholesky(Sigma)
  y_flat <- torch_matmul(L, torch_randn(n_samples, 256, 1))
  y <- torch_reshape(y_flat, c(n_samples, n_grid, n_grid))

  list(
    target = torch_log(torch_stack(list(ell, sigma), 2)),
    conditioning = y
  )
}

test_set <- generate_conditional_samples()
str(test_set)
#> List of 2
#>  $ target      :Float [1:1024, 1:2]
#>  $ conditioning:Float [1:1024, 1:16, 1:16]
par(mfrow = c(2, 2))
for (i in 1 : 4) {
  image(as_array(test_set$conditioning[i, , ]), main = '', xlab = 'x', ylab = 'y', asp = 1)
}
```

![](torchflow_files/figure-html/unnamed-chunk-24-1.png)

We can now create a summarizing network for this conditioning variable.
A convolutional network is a conventional choice for this type of data.
The network alternates between convolution, ReLU and max pooling layers,
with a final adaptive average pooling layer to reduce the summary
statistics to a fixed size vector of dimension 32 (the number of summary
statistics used by the flow):

``` r

summary_model <- nn_sequential(
  # Adds a unit dimension for the channel
  nn_unflatten(2, c(1, n_grid)),
  nn_conv2d(1, 16, 3, padding = 1),
  nn_relu(),
  nn_max_pool2d(2),
  nn_conv2d(16, 32, 3, padding = 1),
  nn_relu(),
  # This averages over the grid to produce a vector of summary statistics
  nn_adaptive_avg_pool2d(1),
  nn_flatten()
)

str(summary_model(test_set$conditioning[1 : 10, ]))
#> Float [1:10, 1:32]
```

We can now create a conditional flow with this summarizing network:

``` r

flow_model <- nn_sequential_conditional_flow(
  nn_affine_coupling_block(2, 32),
  nn_permutation_flow(2),
  nn_affine_coupling_block(2, 32),
  nn_permutation_flow(2),
  nn_affine_coupling_block(2, 32)
)
summarizing_flow_model <- nn_summarizing_conditional_flow(summary_model, flow_model)
```

We can now train the model as before:

``` r

train_conditional_flow(
  summarizing_flow_model,
  generate_conditional_samples,
  n_epochs = 128,
  batch_size = 1024,
  after_epoch = function(...) {
    test_loss <- as_array(forward_kl_loss(summarizing_flow_model(test_set$target, test_set$conditioning)))
    cat('Test loss:', test_loss, '\n')
  }
)
#> = Starting epoch 1 
#> Test loss: 1.241299 
#> = Starting epoch 2 
#> Test loss: 1.213195 
#> = Starting epoch 3 
#> Test loss: 1.181334 
#> = Starting epoch 4 
#> Test loss: 1.145761 
#> = Starting epoch 5 
#> Test loss: 1.10692 
#> = Starting epoch 6 
#> Test loss: 1.065282 
#> = Starting epoch 7 
#> Test loss: 1.022015 
#> = Starting epoch 8 
#> Test loss: 0.9786137 
#> = Starting epoch 9 
#> Test loss: 0.9378191 
#> = Starting epoch 10 
#> Test loss: 0.9002584 
#> = Starting epoch 11 
#> Test loss: 0.8659097 
#> = Starting epoch 12 
#> Test loss: 0.831512 
#> = Starting epoch 13 
#> Test loss: 0.7923454 
#> = Starting epoch 14 
#> Test loss: 0.7496518 
#> = Starting epoch 15 
#> Test loss: 0.7023552 
#> = Starting epoch 16 
#> Test loss: 0.6513882 
#> = Starting epoch 17 
#> Test loss: 0.5973183 
#> = Starting epoch 18 
#> Test loss: 0.5409579 
#> = Starting epoch 19 
#> Test loss: 0.4831415 
#> = Starting epoch 20 
#> Test loss: 0.4240643 
#> = Starting epoch 21 
#> Test loss: 0.3644621 
#> = Starting epoch 22 
#> Test loss: 0.3034844 
#> = Starting epoch 23 
#> Test loss: 0.2430431 
#> = Starting epoch 24 
#> Test loss: 0.1839069 
#> = Starting epoch 25 
#> Test loss: 0.1270038 
#> = Starting epoch 26 
#> Test loss: 0.06826788 
#> = Starting epoch 27 
#> Test loss: 0.004615724 
#> = Starting epoch 28 
#> Test loss: -0.05956089 
#> = Starting epoch 29 
#> Test loss: -0.1031048 
#> = Starting epoch 30 
#> Test loss: -0.125881 
#> = Starting epoch 31 
#> Test loss: -0.1800785 
#> = Starting epoch 32 
#> Test loss: -0.2489877 
#> = Starting epoch 33 
#> Test loss: -0.3175726 
#> = Starting epoch 34 
#> Test loss: -0.3519652 
#> = Starting epoch 35 
#> Test loss: -0.3826392 
#> = Starting epoch 36 
#> Test loss: -0.4236596 
#> = Starting epoch 37 
#> Test loss: -0.4397382 
#> = Starting epoch 38 
#> Test loss: -0.4628551 
#> = Starting epoch 39 
#> Test loss: -0.5020535 
#> = Starting epoch 40 
#> Test loss: -0.5605165 
#> = Starting epoch 41 
#> Test loss: -0.5864271 
#> = Starting epoch 42 
#> Test loss: -0.6153975 
#> = Starting epoch 43 
#> Test loss: -0.6466674 
#> = Starting epoch 44 
#> Test loss: -0.6606249 
#> = Starting epoch 45 
#> Test loss: -0.6904659 
#> = Starting epoch 46 
#> Test loss: -0.7522526 
#> = Starting epoch 47 
#> Test loss: -0.8088346 
#> = Starting epoch 48 
#> Test loss: -0.8068212 
#> = Starting epoch 49 
#> Test loss: -0.8952706 
#> = Starting epoch 50 
#> Test loss: -0.9282489 
#> = Starting epoch 51 
#> Test loss: -0.784091 
#> = Starting epoch 52 
#> Test loss: -0.9843494 
#> = Starting epoch 53 
#> Test loss: -0.9552588 
#> = Starting epoch 54 
#> Test loss: -0.9104232 
#> = Starting epoch 55 
#> Test loss: -1.007158 
#> = Starting epoch 56 
#> Test loss: -1.062367 
#> = Starting epoch 57 
#> Test loss: -1.075744 
#> = Starting epoch 58 
#> Test loss: -1.119204 
#> = Starting epoch 59 
#> Test loss: -1.092405 
#> = Starting epoch 60 
#> Test loss: -1.17273 
#> = Starting epoch 61 
#> Test loss: -1.20098 
#> = Starting epoch 62 
#> Test loss: -1.223645 
#> = Starting epoch 63 
#> Test loss: -1.257624 
#> = Starting epoch 64 
#> Test loss: -1.258504 
#> = Starting epoch 65 
#> Test loss: -1.240636 
#> = Starting epoch 66 
#> Test loss: -1.284702 
#> = Starting epoch 67 
#> Test loss: -1.356329 
#> = Starting epoch 68 
#> Test loss: -1.37797 
#> = Starting epoch 69 
#> Test loss: -1.388709 
#> = Starting epoch 70 
#> Test loss: -1.441272 
#> = Starting epoch 71 
#> Test loss: -1.38496 
#> = Starting epoch 72 
#> Test loss: -1.446184 
#> = Starting epoch 73 
#> Test loss: -1.403917 
#> = Starting epoch 74 
#> Test loss: -1.448674 
#> = Starting epoch 75 
#> Test loss: -1.422456 
#> = Starting epoch 76 
#> Test loss: -1.497867 
#> = Starting epoch 77 
#> Test loss: -1.519677 
#> = Starting epoch 78 
#> Test loss: -1.482682 
#> = Starting epoch 79 
#> Test loss: -1.446197 
#> = Starting epoch 80 
#> Test loss: -1.493649 
#> = Starting epoch 81 
#> Test loss: -1.628282 
#> = Starting epoch 82 
#> Test loss: -1.497611 
#> = Starting epoch 83 
#> Test loss: -1.096736 
#> = Starting epoch 84 
#> Test loss: -1.570871 
#> = Starting epoch 85 
#> Test loss: -1.486406 
#> = Starting epoch 86 
#> Test loss: -1.4958 
#> = Starting epoch 87 
#> Test loss: -1.544404 
#> = Starting epoch 88 
#> Test loss: -1.554627 
#> = Starting epoch 89 
#> Test loss: -1.488919 
#> = Starting epoch 90 
#> Test loss: -1.460143 
#> = Starting epoch 91 
#> Test loss: -1.614595 
#> = Starting epoch 92 
#> Test loss: -1.609903 
#> = Starting epoch 93 
#> Test loss: -1.635557 
#> = Starting epoch 94 
#> Test loss: -1.639202 
#> = Starting epoch 95 
#> Test loss: -1.702752 
#> = Starting epoch 96 
#> Test loss: -1.626896 
#> = Starting epoch 97 
#> Test loss: -1.653559 
#> = Starting epoch 98 
#> Test loss: -1.683834 
#> = Starting epoch 99 
#> Test loss: -1.737577 
#> = Starting epoch 100 
#> Test loss: -1.687341 
#> = Starting epoch 101 
#> Test loss: -1.780876 
#> = Starting epoch 102 
#> Test loss: -1.754284 
#> = Starting epoch 103 
#> Test loss: -1.797385 
#> = Starting epoch 104 
#> Test loss: -1.781355 
#> = Starting epoch 105 
#> Test loss: -1.795308 
#> = Starting epoch 106 
#> Test loss: -1.822464 
#> = Starting epoch 107 
#> Test loss: -1.80223 
#> = Starting epoch 108 
#> Test loss: -1.874405 
#> = Starting epoch 109 
#> Test loss: -1.810679 
#> = Starting epoch 110 
#> Test loss: -1.849463 
#> = Starting epoch 111 
#> Test loss: -1.879088 
#> = Starting epoch 112 
#> Test loss: -1.855226 
#> = Starting epoch 113 
#> Test loss: -1.861955 
#> = Starting epoch 114 
#> Test loss: -1.886285 
#> = Starting epoch 115 
#> Test loss: -1.767466 
#> = Starting epoch 116 
#> Test loss: -1.855772 
#> = Starting epoch 117 
#> Test loss: -1.911113 
#> = Starting epoch 118 
#> Test loss: -1.817192 
#> = Starting epoch 119 
#> Test loss: -1.847489 
#> = Starting epoch 120 
#> Test loss: -1.93243 
#> = Starting epoch 121 
#> Test loss: -1.914279 
#> = Starting epoch 122 
#> Test loss: -1.866788 
#> = Starting epoch 123 
#> Test loss: -1.889521 
#> = Starting epoch 124 
#> Test loss: -1.965211 
#> = Starting epoch 125 
#> Test loss: -1.77689 
#> = Starting epoch 126 
#> Test loss: -1.664507 
#> = Starting epoch 127 
#> Test loss: -1.977085 
#> = Starting epoch 128 
#> Test loss: -1.80753
```

We can now generate samples from the trained model:

``` r

test_samples <- as_array(generate_from_conditional_flow(summarizing_flow_model, 1024, test_set$conditioning[1 : 4, , , drop = FALSE]))
str(test_samples)
#>  num [1:1024, 1:4, 1:2] -0.7415 -0.8741 -0.83986 0.00521 -0.55005 ...

test_target <- as_array(test_set$target)

par(mfrow = c(4, 2))
for (i in 1 : 4) {
  hist(exp(test_samples[, i, 1]), main = '', xlab = 'ell', freq = FALSE, breaks = 32, xlim = c(0, 2))
  abline(v = exp(test_target[i, 1]), col = 'red')
  hist(exp(test_samples[, i, 2]), main = '', xlab = 'sigma', freq = FALSE, breaks = 32, xlim = c(0, 3))
  abline(v = exp(test_target[i, 2]), col = 'red')
}
```

![](torchflow_files/figure-html/unnamed-chunk-28-1.png)

We can also plot the samples on a scatter plot:

``` r

par(mfrow = c(2, 2))
for (i in 1 : 4) {
  plot(exp(test_samples[, i, 1]), exp(test_samples[, i, 2]), main = '', xlab = 'ell', ylab = 'sigma', xlim = c(0, 2), ylim = c(0, 3))
  abline(v = exp(test_target[i, 1]), col = 'red')
  abline(h = exp(test_target[i, 2]), col = 'red')
}
```

![](torchflow_files/figure-html/unnamed-chunk-29-1.png)
