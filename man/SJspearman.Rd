\name{SJspearman}
\alias{SJspearman}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Simulate joint given marginals and Spearman correlations.
}
\description{
Reorder elements in each column of a matrix such that the column-wise Spearman correlations approximate a given correlation matrix.
}
\usage{
SJspearman(
  X,
  cor,
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
\item{cor }{Spearman correlation matrix of \code{X}.}
}
\examples{
# =============================================================================
# Use the same example from <https://cran.r-project.org/web/packages/
#                            SimMultiCorrData/vignettes/workflow.html>.
# =============================================================================
set.seed(123)
N = 10000L # Sample size.
K = 10L # 10 marginals.
# Sample from 3 PDFs, 2 nonparametric PMFs, 5 parametric PMFs:
marginals = cbind(
  rnorm(N), rchisq(N, 4), rbeta(N, 4, 2),
  LHSpmf(data.frame(val = 1:3, P = c(0.3, 0.45, 0.25)), N,
         seed = sample(1e6L, 1)),
  LHSpmf(data.frame(val = 1:4, P = c(0.2, 0.3, 0.4, 0.1)), N,
         seed = sample(1e6L, 1)),
  rpois(N, 1), rpois(N, 5), rpois(N, 10),
  rnbinom(N, 3, 0.2), rnbinom(N, 6, 0.8))
# The seeding for `LHSpmf()` is unhealthy, but OK for small examples.


marginals = apply(marginals, 2, function(x) sort(x))


# Create the target correlation matrix `Rey` treated as Spearman
# correlations.
set.seed(11)
Rey <- diag(1, nrow = 10)
for (i in 1:nrow(Rey)) {
  for (j in 1:ncol(Rey)) {
    if (i > j) Rey[i, j] <- runif(1, 0.2, 0.7)
    Rey[j, i] <- Rey[i, j]
  }
}


result = SimJoint::SJspearman(
  X = marginals, cor = Rey, errorType = "meanSquare", seed = 456,
  maxCore = 1, convergenceTail = 8, verbose = TRUE)


# Check relative errors.
summary(as.numeric(abs(cor(result$X, method = "spearman") / Rey - 1)))




# Another way to impose rank correlation is to supply rank matrix
# to SJpearson():
system.time({reorderedRanks = SimJoint::SJpearson(
  X = apply(marginals, 2, function(x) rank(x)), cor = Rey,
  errorType = "meanSquare", seed = 456, maxCore = 1,
  convergenceTail = 8, verbose = TRUE)})


# Reordering according to ranks:
result = apply(rbind(reorderedRanks$X, marginals), 2, function(x)
{
  x[(N + 1L) : (2L * N)][as.integer(x[1L : N])]
})


# Check the relative errors.
summary(as.numeric(abs(cor(result, method = "spearman") / Rey - 1)))
}
