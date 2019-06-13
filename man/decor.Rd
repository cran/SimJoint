\name{decor}
\alias{decor}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Create uncorrelated data
}
\description{
Create a matrix where columns are (Pearson) uncorrelated.
}
\usage{
decor(seedMat)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{seedMat}{
A matrix where the number of rows is no less than the number of columns. The function will change \code{seedMat}.
}
}
\details{
Algorithm: for \code{i = 2} to \code{ncol(seedMat)}, the function replaces the first \code{i - 1} elements of the \code{i}th column with values such that the new \code{i}th column becomes uncorrelated with the first \code{i - 1} columns.
}
\value{
None.
}


\examples{
set.seed(123)
X = matrix(rnorm(1000), ncol = 10)
corMat = cor(X)
summary(corMat[corMat < 1]) # Off-diagonal.
# Min.      1st Qu.   Median     Mean  3rd Qu.     Max.
# -0.19271 -0.05648 -0.02272 -0.01303  0.01821  0.24521


SimJoint::decor(X)
corMat2 = cor(X)
summary(corMat2[corMat2 < 1])
# Min.           1st Qu.    Median       Mean    3rd Qu.       Max.
# -2.341e-17 -3.627e-18  3.766e-18  4.018e-18  1.234e-17  3.444e-17
}

