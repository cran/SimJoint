

# =============================================================================
# Use the same example from <https://cran.r-project.org/web/packages/
#                            SimMultiCorrData/vignettes/workflow.html>.
# =============================================================================
rm(list = ls()); gc()
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
  cor = Rey, errorType = "meanSquare", seed = 456, maxCore = 7,
  convergenceTail = 8, verbose = T)})


# Check relative errors.
summary(as.numeric(abs(result$cor / Rey - 1)))


















