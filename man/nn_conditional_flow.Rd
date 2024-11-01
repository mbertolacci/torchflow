% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/conditional-flow.R
\name{nn_conditional_flow}
\alias{nn_conditional_flow}
\title{Conditional Normalizing Flow}
\usage{
nn_conditional_flow()
}
\description{
A conditional normalizing flow is a normalizing flow that takes an additional
conditioning input. This module provides a base class for conditional
normalizing flows.
}
\details{
The base class, \code{nn_conditional_flow}, is an abstract class that provides a
forward and reverse method, as well as a dimension method. Subclasses created
with \code{\link[torch:nn_module]{torch::nn_module()}} should implement these methods. The class is a subclass
of \code{\link[torch:nn_module]{torch::nn_module()}}, and it inherits all of its methods and semantics.
}
\section{Forward method}{

The forward method should return the output and the log determinant of the
Jacobian in the attribute \code{log_jacobian}. Example:

\if{html}{\out{<div class="sourceCode r">}}\preformatted{forward = function(input, conditioning) \{
  output <- ...
  attr(output, 'log_jacobian') <- ...
  output
\}
}\if{html}{\out{</div>}}
}

\section{Reverse method}{


The reverse method should return the inverse of the output, but need not
implement a log determinant. Example:

\if{html}{\out{<div class="sourceCode r">}}\preformatted{reverse = function(input, conditioning) \{
  output <- ...
  output
\}
}\if{html}{\out{</div>}}
}

\section{Dimension method}{


The dimension method should return the dimension of the input and output of
the flow. Example:

\if{html}{\out{<div class="sourceCode r">}}\preformatted{dimension = function() \{
  return(2)
\}
}\if{html}{\out{</div>}}
}

\seealso{
\code{\link[=nn_summarizing_conditional_flow]{nn_summarizing_conditional_flow()}}, \code{\link[=nn_sequential_conditional_flow]{nn_sequential_conditional_flow()}},
\code{\link[=nn_permutation_flow]{nn_permutation_flow()}}, and \code{\link[=nn_affine_coupling_block]{nn_affine_coupling_block()}} for subclasses.
}
