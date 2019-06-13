\name{exportRandomState}
\alias{exportRandomState}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Export Mersenne Twister engine
}
\description{
Export the complete state of a Mersenne Twister engine implemented in C++ 11 standard library <random>.
}
\usage{
exportRandomState(seed)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{seed}{
An integer for seeding a random number generator (RNG) of type \code{std::mt19937_64}.
}
}
\details{
The returned integer vector \code{Value} is analogous to \code{.Random.seed}. \code{Value} can seed for all functions that need a RNG in this package. In those functions, memory address of \code{Value[1]} is assigned to a pointer of type \code{std::mt19937_64}. Realization of this pointer generates random numbers when needed, and \code{Value} changes accordingly.
}
\value{
An integer vector of size 626. It contains all the bits in a \code{std::mt19937_64} object.
}

\examples{
# Make a random PMF.
set.seed(456)
val = seq(0, 15, len = 100)
pmf = data.frame(
  val = val, P = dgamma(val, shape = 2, scale = 2) + runif(100) * 0.1)
pmf$P = pmf$P / sum(pmf$P)


completeRandomState = SimJoint::exportRandomState(456)
# `completeRandomState` comprises all the bits of a Mersenne Twister
# (C++11 std::mt19937_64) engine seeded by 456. It is similar to R's
# `.Random.seed`.
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
