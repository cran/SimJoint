\name{SJpearsonPMF}
\alias{SJpearsonPMF}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Simulate joint with marginal PMFs and Pearson correlations.
}
\description{
Sample from marginal probability mass functions via Latin hypercube sampling and then simulate the joint distribution with Pearson correlations.  Use \code{xSJpearsonPMF()} for the freedom of supplying the noise matrix, which can let the dependency structure of the result joint distribution be characterized by a certain copula. See the copula section in the package vignette for details.
}
\usage{
SJpearsonPMF(
  PMFs,
  sampleSize,
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
  \item{PMFs}{
A list of data frames. Each data frame has 2 columns, a value vector and a probability vector. Probabilities should sum up to 1. Let the size of \code{PMFs} be \code{K}.
}
  \item{sampleSize}{
An integer. The sample size \code{N}.
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


system.time({result = SimJoint::SJpearsonPMF(
  PMFs = PMFs, sampleSize = N, cor = Rey, errorType = "meanSquare",
  seed = 456, maxCore = 2, convergenceTail = 8, verbose = TRUE)})


# Check relative errors.
summary(as.numeric(abs(result$cor / Rey - 1)))




# =============================================================================
# Play with random nonparametric PMFs.
# =============================================================================
set.seed(123)
N = 2000L
K = 20L


# Create totally random nonparametric PMFs:
PMFs = lapply(1L : K, function(x)
{
  p = runif(2, 1, 10)
  result = data.frame(
    val = sort(rnorm(200)), P = runif(200))
  result$P = result$P / sum(result$P)
  result
})


# Create a valid correlation matrix upper-bounded by `frechetUpperCor`.
while(TRUE)
{
  targetCor = matrix(runif(K * K, -0.1, 0.3), ncol = K)
  targetCor[lower.tri(targetCor)] = t(targetCor)[lower.tri(t(targetCor))]
  diag(targetCor) = 1
  if(min(eigen(targetCor)$values) >= 0) break # Break once the correlation
  # matrix is semi-positive definite. This loop could be running for quite
  # a long time if we do not bound `runif()`.
}


result = SimJoint::SJpearsonPMF(
  PMFs = PMFs, sampleSize = N, cor = targetCor, stochasticStepDomain = c(0, 1),
  errorType = "meanSquare", seed = 456, maxCore = 2, convergenceTail = 8)


# Visualize errors and correlation matrices.
par(mfrow = c(2, 2))
hist(result$cor - targetCor, breaks = K * K, main = NULL,
     xlab = "Error", cex.lab = 1.5, cex.axis = 1.25)
hist(result$cor / targetCor - 1, breaks = K * K, main = NULL,
     xlab = "Relative error", ylab = "", cex.lab = 1.5, cex.axis = 1.25)
zlim = range(range(targetCor[targetCor < 1]), range(result$cor[result$cor < 1]))
col = colorRampPalette(c("blue", "red", "yellow"))(K * K)
tmp = targetCor[, K : 1L]
image(tmp, xaxt = "n", yaxt = "n", zlim = zlim, bty = "n",
      main = "Target cor", col = col)
tmp = result$cor[, K : 1L]
image(tmp, xaxt = "n", yaxt = "n", zlim = zlim, bty = "n",
      main = "Cor reached", col = col)
par(mfrow = c(1, 1))
}
