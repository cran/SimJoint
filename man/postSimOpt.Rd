\name{postSimOpt}
\alias{postSimOpt}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Post simulation optimization
}
\description{
Impose the target correlation matrix via a heuristic algorithm.
}
\usage{
postSimOpt(
  X,
  cor,
  Xcor = matrix(),
  acceptProb = 1,
  seed = 123L,
  convergenceTail = 10000L
  )
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X}{
An \code{N x K} numeric matrix of \code{K} marginal distributions (samples). Columns need not be sorted.
}
  \item{cor}{
A \code{K x K} target correlation matrix. The matrix should be positive semi-definite.
}
  \item{Xcor}{
The \code{K x K} correlation matrix of \code{X}. If empty, calculate the correlations inside. Default empty.
}
  \item{acceptProb}{
A numeric vector of probabilities that sum up to 1. In each iteration, the entry having the largest error in the current correlation matrix will be selected with probability \code{acceptProb[1]} for correction; the entry having the second largest error will be selected with probability \code{acceptProb[2]} for correction, etc. Default 1, meaning the entry with the worst error is always chosen.
}
  \item{seed}{
An integer or an integer vector of size 4. A single integer seeds a \code{pcg64} generator the usual way. An integer vector of size 4 supplies all the bits for a \code{pcg64} object. Default 123.
}
  \item{convergenceTail}{
An integer. If the last \code{convergenceTail} iterations did not reduce the cost function, return. Default 100000.
}
}
\details{
Algorithms are detailed in the package vignette. Examples of usage also appeared in functions like \code{SJpearson()}.
}
\value{
A list of size 2.
\item{X }{A numeric matrix of size \code{N x K}, the simulated joint distribution.}
\item{cor }{Pearson correlation matrix of \code{X}.}
}
\examples{
# =============================================================================
# Use one of the examples for `SJpearson()`
# =============================================================================
set.seed(123)
N = 10000L
K = 10L


# Several 2-parameter PDFs in R:
marginals = list(rbeta, rcauchy, rf, rgamma, rnorm, runif, rweibull)
Npdf = length(marginals)


if(Npdf >= K) chosenMarginals =
  marginals[sample(Npdf, K, replace = TRUE)] else chosenMarginals =
  marginals[c(1L : Npdf, sample(Npdf, K - Npdf, replace = TRUE))]


# Sample from the marginal PDFs.
marginals = as.matrix(as.data.frame(lapply(chosenMarginals, function(f)
{
  para = sort(runif(2, 0.1, 10))
  rst = f(N, para[1], para[2])
  sort(rst)
})))
dimnames(marginals) = NULL


frechetUpperCor = cor(marginals) # The correlation matrix should be
# upper-bounded by that of the perfectly rank-correlated
# joint (Frechet upper bound). The lower bound is characterized by
# d-countercomonotonicity and depends not only on marginals.
cat("Range of maximal correlations between marginals:",
    range(frechetUpperCor[frechetUpperCor < 1]))
# Two perfectly rank-correlated marginals can have a Pearson
# correlation below 0.07. This is due to highly nonlinear functional
# relationships between marginal PDFs.


# Create a valid correlation matrix upper-bounded by `frechetUpperCor`.
while(TRUE)
{
  targetCor = sapply(frechetUpperCor, function(x)
    runif(1, -0.1, min(0.3, x * 0.8)))
  targetCor = matrix(targetCor, ncol = K)
  targetCor[lower.tri(targetCor)] = t(targetCor)[lower.tri(t(targetCor))]
  diag(targetCor) = 1
  if(min(eigen(targetCor)$values) >= 0) break # Stop once the correlation
  # matrix is semi-positive definite. This loop could run for
  # a long time if we do not bound the uniform by 0.3.
}


result = SimJoint::SJpearson(
  X = marginals, cor = targetCor, stochasticStepDomain = c(0, 1),
  errorType = "meanSquare", seed = 456, maxCore = 1, convergenceTail = 8)

# # Code blocks are commented due to execution time constraint by CRAN check.
# system.time({postOptResult = SimJoint::postSimOpt(
#   X = result$X, cor = targetCor, convergenceTail = 10000)})
# # user  system elapsed
# # 6.66    0.00    6.66
#
# system.time({directOptResult = SimJoint::postSimOpt(
#   X = marginals, cor = targetCor, convergenceTail = 10000)})
# # user  system elapsed
# # 8.48    0.00    8.48
#
# sum((result$cor - targetCor) ^ 2)
# # [1] 0.02209447
# sum((resultOpt$cor - targetCor) ^ 2)
# # [1] 0.0008321346
# sum((directOptResult$cor - targetCor) ^ 2)
# # [1] 0.02400257
}








