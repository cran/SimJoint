

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










