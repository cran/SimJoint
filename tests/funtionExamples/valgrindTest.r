

LD_PRELOAD="/lib64/libasan.so.5 /lib64/libubsan.so.1"  R


# R -d valgrind



wget https://sourceware.org/pub/valgrind/valgrind-3.22.0.tar.bz2
tar -xjf valgrind-3.22.0.tar.bz2
cd valgrind-3.22.0
./configure --prefix=/usr/local
make
sudo make install
ccache --clear




R -d /usr/local/bin/valgrind
# R -d valgrind
setwd('/finance_develop/Charlie/simjointTest/SimJoint')
file.copy('/finance_develop/Charlie/vipCodes/cpp/Charlie/ThreadPool.hpp',
          '/finance_develop/Charlie/simjointTest/SimJoint/src/hpp', overwrite = T)


source('../../vipCodes/R/CharlieSourceCpp.R')
CharlieSourceCpp('src/jointSimulation.cpp', optFlag = '-O0', sanitize = F,
                 cppStd = '-std=gnu++17', compilerPath = 'g++', rebuild = T)
CharlieSourceCpp('src/lhsPMF.cpp', optFlag = '-O0', sanitize = F,
                 cppStd = '-std=gnu++17', compilerPath = 'g++', rebuild = T)


# =============================================================================
# Benchmark against R package `SimMultiCorrData`. Use the same example
# from <https://cran.r-project.org/web/packages/SimMultiCorrData/
#       vignettes/workflow.html>.
# =============================================================================
set.seed(123)
N = 10000L # Sample size.
K = 10L    # 10 marginals.
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


# Create the example target correlation matrix `Rey`:
set.seed(11)
Rey <- diag(1, nrow = K)
for (i in 1:nrow(Rey)) {
  for (j in 1:ncol(Rey)) {
    if (i > j) Rey[i, j] <- runif(1, 0.2, 0.7)
    Rey[j, i] <- Rey[i, j]
  }
}


system.time({result = SJpearson(
  X = marginals, cor = Rey, errorType = "meanSquare", seed = 456,
  maxCore = 1, convergenceTail = 8, verbose = FALSE)})
# user  system elapsed
# 0.30    0.00    0.29
# One the same platform, single-threaded speed (Intel i7-4770 CPU
# @ 3.40GHz, 32GB RAM, Windows 10, g++ 4.9.3 -Ofast, R 3.5.2) is more
# than 50 times faster than `SimMultiCorrData::rcorrvar()`:
# user  system elapsed
# 16.05   0.34   16.42


# Check error statistics.
summary(as.numeric(round(cor(result$X) - Rey, 6)))
# Min.          1st Qu.     Median       Mean    3rd Qu.       Max.
# -0.000365   -0.000133  -0.000028  -0.000047  0.000067    0.000301


# Post simulation optimization further reduce the errors:
resultOpt = postSimOpt(
  X = result$X, cor = Rey, convergenceTail = 10)
summary(as.numeric(round(cor(resultOpt$X) - Rey, 6)))
# Min.        1st Qu.    Median      Mean   3rd Qu.      Max.
# -7.10e-05 -3.10e-05 -1.15e-05 -6.48e-06  9.00e-06  7.10e-05


# Max error magnitude is less than 1% of that from
# `SimMultiCorrData::rcorrvar()`:
# Min.          1st Qu.     Median       Mean    3rd Qu.       Max.
# -0.008336   -0.001321          0  -0.000329   0.001212    0.00339
# This table is reported in Step 4, correlation methods 1 or 2.




# =============================================================================
# Use the above example and benchmark against John Ruscio & Walter
# Kaczetow (2008) iteration. The R code released with their paper was
# erroneous. A corrected version is given by Github user "nicebread":
# <https://gist.github.com/nicebread/4045717>, but his correction was
# incomprehensive and can only handle 2-dimensional instances. Please change
# Line 32 to `Target.Corr <- rho` and source the file.
# =============================================================================
# # Test Ruscio-Kaczetow's code.
# set.seed(123)
# RuscioKaczetow = GenData(Pop = as.data.frame(marginals),
#                          Rey, N = 1000) # By default, the function takes 1000
# # samples from each marginal population of size 10000.
# summary(round(as.numeric(cor(RuscioKaczetow) - Rey), 6))
# # Min.         1st Qu.    Median      Mean   3rd Qu.      Max.
# # -0.183274 -0.047461  -0.015737 -0.008008  0.027475  0.236662


result = SJpearson(
  X = apply(marginals, 2, function(x) sort(sample(x, 1000, replace = TRUE))),
  cor = Rey, errorType = "maxRela", maxCore = 1000) # CRAN does not allow more
# than 2 threads for running examples.
summary(round(as.numeric(cor(result$X) - Rey), 6))
# Min.           1st Qu.     Median       Mean    3rd Qu.       Max.
# -0.0055640  -0.0014850 -0.0004810 -0.0007872  0.0000000  0.0025920
resultOpt = postSimOpt(
  X = result$X, cor = Rey, convergenceTail = 10000)
summary(as.numeric(round(cor(resultOpt$X) - Rey, 6)))
# Min.          1st Qu.     Median       Mean    3rd Qu.       Max.
# -6.240e-04 -2.930e-04 -2.550e-05 -6.532e-05  1.300e-04  5.490e-04

