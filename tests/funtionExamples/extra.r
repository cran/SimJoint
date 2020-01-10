

# Make a random PMF.
set.seed(456)
val = seq(0, 15, len = 100)
pmf = data.frame(
  val = val, P = dgamma(val, shape = 2, scale = 2) + runif(100) * 0.1)
pmf$P = pmf$P / sum(pmf$P)


completeRandomState = SimJoint::exportRandomState(456)
# `completeRandomState` comprises all the bits of a pcg64
# engine seeded by 456. It is similar to R's `.Random.seed`.
# sink('debug.txt')
completeRandomState
pmfSample1 = SimJoint::LHSpmf(pmf, 1000, completeRandomState)
completeRandomState
pmfSample2 = SimJoint::LHSpmf(pmf, 1000, completeRandomState)
completeRandomState
pmfSample3 = SimJoint::LHSpmf(pmf, 1000, completeRandomState)
completeRandomState
# `completeRandomState` is changed in each run of `LHSpmf()`.
# sink()

















