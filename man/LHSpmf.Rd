\name{LHSpmf}
\alias{LHSpmf}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Sample from probability mass function
}
\description{
Sample from a probability mass function (PMF) via Latin hypercube sampling.
}
\usage{
LHSpmf(pmf, sampleSize, seed)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{pmf}{
A 2-column data frame as a PMF. The 1st column is sorted and contains value points. The 2nd column contains probabilities. Probabilities should sum up to 1.
}
  \item{sampleSize}{
Sample size.
}
  \item{seed}{
An integer vector of size 1 or 626. Both seed a Mersenne Twister RNG while the latter gives the complete state of the RNG.
}
}

\value{
Random samples from \code{pmf} as a numeric vector of size \code{sampleSize}.
}


\examples{
# Make a random PMF.
val = seq(0, 15, len = 100)
pmf = data.frame(val = val, P = dgamma(val, shape = 2, scale = 2))
pmf$P = pmf$P / sum(pmf$P)
pmfSample = SimJoint::LHSpmf(pmf, 1000, 123)
hist(pmfSample, breaks = 200)
}
