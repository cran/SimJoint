

set.seed(123)
X = matrix(rnorm(1000), ncol = 10)
corMat = cor(X)
summary(corMat[corMat < 1])
# Min.      1st Qu.   Median     Mean  3rd Qu.     Max.
# -0.19271 -0.05648 -0.02272 -0.01303  0.01821  0.24521


SimJoint::deCorSimDat(X)
corMat2 = cor(X)
summary(corMat2[corMat2 < 1])
# Min.           1st Qu.    Median       Mean    3rd Qu.       Max.
# -2.341e-17 -3.627e-18  3.766e-18  4.018e-18  1.234e-17  3.444e-17












