\name{xSJpearson}
\alias{xSJpearson}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Simulate joint given marginals, Pearson correlations and uncorrelated support matrix.
}
\description{
Users specify the uncorrelated random source instead of using a permuted \code{X} to left-multiply the correlation matrix decomposition. See the package vignette for more details.
}
\usage{
xSJpearson(
  X,
  cor,
  noise,
  stochasticStepDomain = as.numeric(c(0, 1)),
  errorType = "meanSquare",
  seed = 123L,
  maxCore = 7L,
  convergenceTail = 8L,
  iterLimit = 100000L,
  verbose = TRUE
  )
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X}{
An \code{N x K} numeric matrix of \code{K} marginal distributions (samples). Columns are sorted.
}
  \item{cor}{
A \code{K x K} correlation matrix. The matrix should be positive semi-definite.
}
  \item{noise}{
An \code{N x K} arbitrary numeric matrix where columns are (more or less) uncorrelated. Exact zero correlations are unnecessary.
}
  \item{stochasticStepDomain}{
A numeric vector of size 2. Range of the stochastic step ratio for correcting the correlation matrix in each iteration. Default [0, 1]. See the package vignette for more details.
}
  \item{errorType}{
Cost function for convergence test.

\code{"meanRela"}: average absolute relative error between elements of the target correlation matrix and the correlation matrix approximated in each iteration.

\code{"maxRela"}: maximal absolute relative error.

\code{"meanSquare"}: mean squared error. Default.
}
  \item{seed}{
An integer or an integer vector of size 626. A single integer seeds a \code{std::mt19937_64} (C++11 Mersenne Twister) generator the usual way. An integer vector of size 626 supplies all the bits for a \code{std::mt19937_64} object.
}
  \item{maxCore}{
An integer. Maximal threads to invoke. Default 7. Better be no greater than the total number of virtual cores on machine.
}
  \item{convergenceTail}{
An integer. If the last \code{convergenceTail} iterations resulted in equal cost function values, return. Default 8.
}
  \item{iterLimit}{
An integer. The maximal number of iterations. Default 100000.
}
  \item{verbose}{
A boolean value. \code{TRUE} prints progress.
}
}
\details{
Algorithms are detailed in the package vignette.
}
\value{
A list of size 2.
\item{X }{A numeric matrix of size \code{N x K}, the simulated joint distribution.}
\item{cor }{Pearson correlation matrix of \code{X}.}
}

\examples{
# =============================================================================
# Use the same example from <https://cran.r-project.org/web/packages/
#                            SimMultiCorrData/vignettes/workflow.html>.
# =============================================================================
rm(list = ls()); gc()
set.seed(123)
N = 10000L # Sample size.
K = 10L # 10 marginals.
# Sample from 3 PDFs, 2 nonparametric PMFs, 5 parametric PMFs:
marginals = cbind(
  rnorm(N), rchisq(N, 4), rbeta(N, 4, 2),
  SimJoint::LHSpmf(data.frame(val = 1:3, P = c(0.3, 0.45, 0.25)), N,
         seed = sample(1e6L, 1)),
  SimJoint::LHSpmf(data.frame(val = 1:4, P = c(0.2, 0.3, 0.4, 0.1)), N,
         seed = sample(1e6L, 1)),
  rpois(N, 1), rpois(N, 5), rpois(N, 10),
  rnbinom(N, 3, 0.2), rnbinom(N, 6, 0.8))
# The seeding for `LHSpmf()` is unhealthy, but OK for small examples.


marginals = apply(marginals, 2, function(x) sort(x))


# Create the target correlation matrix `Rey`:
set.seed(11)
Rey <- diag(1, nrow = K)
for (i in 1:nrow(Rey)) {
  for (j in 1:ncol(Rey)) {
    if (i > j) Rey[i, j] <- runif(1, 0.2, 0.7)
    Rey[j, i] <- Rey[i, j]
  }
}


system.time({result = SimJoint::xSJpearson(
  X = marginals, cor = Rey, noise = matrix(runif(N * K), ncol = K),
  errorType = "meanSquare", seed = 456, maxCore = 2,
  convergenceTail = 8, verbose = TRUE)})


summary(as.numeric(round(cor(result$X) - Rey, 6)))
}
