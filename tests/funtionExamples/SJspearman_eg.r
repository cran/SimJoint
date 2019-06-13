

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
  maxCore = 7, convergenceTail = 8, verbose = T)


# Check relative errors.
summary(as.numeric(abs(cor(result$X, method = "spearman") / Rey - 1)))




# Another way to impose rank correlation is to supply rank matrix
# to SJpearson():
system.time({reorderedRanks = SimJoint::SJpearson(
  X = apply(marginals, 2, function(x) rank(x)), cor = Rey,
  errorType = "meanSquare", seed = 456, maxCore = 7,
  convergenceTail = 8, verbose = T)})


# Reordering according to ranks:
result = apply(rbind(reorderedRanks$X, marginals), 2, function(x)
{
  x[(N + 1L) : (2L * N)][as.integer(x[1L : N])]
})


# Check the relative errors.
summary(as.numeric(abs(cor(result, method = "spearman") / Rey - 1)))






















