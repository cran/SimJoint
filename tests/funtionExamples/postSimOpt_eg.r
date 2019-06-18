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


result = SJMC::SJpearson(
  X = marginals, cor = targetCor, stochasticStepDomain = c(0, 1),
  errorType = "meanSquare", seed = 456, maxCore = 2, convergenceTail = 8)

\donttest{
system.time({resultOpt = SJMC::postSimOpt(
  X = result$X, cor = targetCor, convergenceTail = 10000)})
# user  system elapsed
# 6.66    0.00    6.66

system.time({directOptResult = SJMC::postSimOpt(
  X = marginals, cor = targetCor, convergenceTail = 10000)})
# user  system elapsed
# 8.48    0.00    8.48

sum((result$cor - targetCor) ^ 2)
# [1] 0.02209447
sum((resultOpt$cor - targetCor) ^ 2)
# [1] 0.0008321346
sum((directOptResult$cor - targetCor) ^ 2)
# [1] 0.02400257
}











