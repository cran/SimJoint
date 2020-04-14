\name{xSJpearsonPMF}
\alias{xSJpearsonPMF}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Simulate joint with marginal PMFs, Pearson correlations and uncorrelated support matrix.}
\description{
Sample from marginal probability mass functions via Latin hypercube sampling and then simulate the joint distribution with Pearson correlations. Users specify the uncorrelated random source instead of using permuted marginal samples to left-multiply the correlation matrix decomposition.
}
\usage{
xSJpearsonPMF(
  PMFs,
  sampleSize,
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
  \item{PMFs}{
A list of data frames. Each data frame has 2 columns, a value vector and a probability vector. Probabilities should sum up to 1. Let the size of \code{PMFs} be \code{K}.
}
  \item{sampleSize}{
An integer. The sample size \code{N}.
}
  \item{cor}{
A \code{K x K} positive semi-definite correlation matrix.
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
An integer or an integer vector of size 4. A single integer seeds a \code{pcg64} generator the usual way. An integer vector of size 4 supplies all the bits for a \code{pcg64} object.
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
set.seed(123)
N = 10000L # Sample size.
K = 10L # 10 marginals.
# 3 PDFs, 2 nonparametric PMFs, 5 parametric PMFs:
PMFs = c(
  apply(cbind(rnorm(N), rchisq(N, 4), rbeta(N, 4, 2)), 2, function(x)
    data.frame(val = sort(x), P = 1.0 / N)),
  list(data.frame(val = 1:3 + 0.0, P = c(0.3, 0.45, 0.25))),
  list(data.frame(val = 1:4 + 0.0, P = c(0.2, 0.3, 0.4, 0.1))),
  apply(cbind(rpois(N, 1), rpois(N, 5), rpois(N, 10),
              rnbinom(N, 3, 0.2), rnbinom(N, 6, 0.8)), 2, function(x)
                data.frame(val = as.numeric(sort(x)), P = 1.0 / N))
)


# Create the target correlation matrix `Rey`:
set.seed(11)
Rey <- diag(1, nrow = 10)
for (i in 1:nrow(Rey)) {
  for (j in 1:ncol(Rey)) {
    if (i > j) Rey[i, j] <- runif(1, 0.2, 0.7)
    Rey[j, i] <- Rey[i, j]
  }
}


system.time({result = SimJoint::xSJpearsonPMF(
  PMFs = PMFs, sampleSize = N, noise = matrix(runif(N * K), ncol = K),
  cor = Rey, errorType = "meanSquare", seed = 456, maxCore = 1,
  convergenceTail = 8, verbose = TRUE)})


# Check relative errors.
summary(as.numeric(abs(result$cor / Rey - 1)))
}
