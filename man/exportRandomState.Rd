\name{exportRandomState}
\alias{exportRandomState}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Export Permuted Congruential Generator
}
\description{
Export all the bits needed for seeding Permuted Congruential Generator.
}
\usage{
exportRandomState(seed)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{seed}{
An integer or an integer vector of size 4 (128 bits). See

<http://www.pcg-random.org/>.
}
}
\details{
The returned integer vector \code{Value} supplies all the bits necessary for determining the state of a \code{pcg64} generator. \code{Value} can seed for all functions that need a RNG in this package. It will change after the function call, ready for seeding the \code{pcg64} generator in the next function call.
}
\value{
An integer vector of size 4.
}

\examples{
# Make a random PMF.
set.seed(456)
val = seq(0, 15, len = 100)
pmf = data.frame(
  val = val, P = dgamma(val, shape = 2, scale = 2) + runif(100) * 0.1)
pmf$P = pmf$P / sum(pmf$P)


completeRandomState = SimJoint::exportRandomState(456)
# `completeRandomState` comprises all the bits of a pcg64
# engine seeded by 456. It is similar to R's `.Random.seed`.
pmfSample1 = SimJoint::LHSpmf(pmf, 1000, completeRandomState)
pmfSample2 = SimJoint::LHSpmf(pmf, 1000, completeRandomState)
pmfSample3 = SimJoint::LHSpmf(pmf, 1000, completeRandomState)
# `completeRandomState` is changed in each run of `LHSpmf()`.


targetCor = rbind(
c(1,   0.3, 0.5),
c(0.3,   1, 0.3),
c(0.5, 0.3,   1))


result = SimJoint::SJpearson(
  X = cbind(sort(pmfSample1), sort(pmfSample2), sort(pmfSample3)),
  cor = targetCor, seed = completeRandomState, errorType = "maxRela")


cor(result$X)
}
