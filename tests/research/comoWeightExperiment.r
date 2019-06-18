



rm(list = ls()); gc()
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


# marginals = as.data.frame(marginals)


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
    runif(1, -0.1, min(0.5, x * 0.8)))
  targetCor = matrix(targetCor, ncol = K)
  targetCor[lower.tri(targetCor)] = t(targetCor)[lower.tri(t(targetCor))]
  diag(targetCor) = 1
  if(min(eigen(targetCor)$values) >= 0) break # Stop once the correlation
  # matrix is semi-positive definite. This loop could run for
  # a long time if we do not bound the uniform by 0.3.
}


marginals = as.data.frame(lapply(marginals, function(x) sample(x, length(x))))


popuVar <- function(x) {mean(x ^ 2) - mean(x) ^ 2}
for(i in 2L : ncol(marginals))
{
  for(u in 1L : 1000L)
  {
    st = 1L
    for(j in 1L : (i - 1L))
    {
      currentVar = popuVar(marginals[[i]] + marginals[[j]])
      iSorted = sort(marginals[[i]])
      jSorted = sort(marginals[[j]])
      comoVar = popuVar(iSorted + jSorted)


      rho = targetCor[i, j]
      iVar = popuVar(marginals[[i]])
      jVar = popuVar(marginals[[j]])
      targetVar = iVar + jVar + 2 * rho * (iVar * jVar) ^ 0.5
      w = (targetVar - currentVar) / (comoVar - currentVar)
      # cat(w, "")
      if(w >= 0)
      {
        n = as.integer(round(N * w))
        ind = st : min(st + n - 1L, N)
        if(length(ind) == 1L) break
        if(ind[1] < ind[2])
        {
          marginals[[i]][ind] = sort(marginals[[i]][ind])[rank(marginals[[j]][ind])]
          st = st + length(ind)
        }
      }
      else
      {
        counterComoVar = popuVar(iSorted + rev(jSorted))
        w = (targetVar - currentVar) / (counterComoVar - currentVar)
        n = as.integer(round(N * w))
        ind = st : min(st + n - 1L, N)
        if(length(ind) == 1L) break
        if(ind[1] < ind[2])
        {
          marginals[[i]][ind] = sort(marginals[[i]][ind])[length(ind) + 1L - rank(marginals[[j]][ind])]
          st = st + length(ind)
        }
      }
    }
    # Shuffle
    ind = sample(N, N)
    for(k in 1L : i) marginals[[k]] = marginals[[k]][ind]
  }

}




result = SJMC::SJpearson(X = as.matrix(as.data.frame(lapply(marginals, function(x) sort(x)))), cor = targetCor, stochasticStepDomain = c(0, 1), errorType = "meanSquare", seed = 456, maxCore = 7, convergenceTail = 8)




postCorrection <- function(X, targetCor)
{
  N = nrow(X)
  K = ncol(X)
  for(iter in 1L : 10000L)
  {

    currentCor = cor(X)
    d = abs(currentCor - targetCor)
    cost = sum(d ^ 2)
    # d = abs(currentCor / targetCor - 1)
    # cost = max(d)


    # worst = sample(K * K, 1, prob = exp(d) / sum(exp(d)))
    worst = which.max(d)
    j = worst %% K
    if(j == 0) {i = as.integer(worst / K); j = K}
    else i = as.integer(worst / K) + 1L


    u = sample(N, 2)
    if(targetCor[i, j] > currentCor[i, j])
    {
      while(diff(X[u, i]) * diff(X[u, j]) >= 0)
      {
        u = sample(N, 2)
      }
    }
    else
    {
      while(diff(X[u, i]) * diff(X[u, j]) <= 0)
      {
        u = sample(N, 2)
      }
    }
    X[u, i] = X[rev(u), i]


    currentCor = cor(X)
    d = abs(currentCor - targetCor)
    newCost = sum(d ^ 2)
    # d = abs(currentCor / targetCor - 1)
    # newCost = max(d)


    if(newCost > cost) X[u, i] = X[rev(u), i]
    else cost = newCost
  }
  X
}


rst = postCorrection(result$X, targetCor)


# b = range(c(range(cor(result$X) / targetCor - 1), range(cor(rst) / targetCor - 1)))
# b = seq(b[1], b[2], len = 100)
# hist(cor(result$X) / targetCor - 1, col = scales::alpha("darkblue", 0.3), border = NA, breaks = b); hist(cor(rst) / targetCor - 1, breaks = 100, add = T, col = scales::alpha("red", 0.3), border = NA)




result = SJMC::SJpearson(X = marginals, cor = targetCor, stochasticStepDomain = c(0, 1), errorType = "meanSquare", seed = 456, maxCore = 7, convergenceTail = 8, verbose = F)


b = range(c(range(cor(result$X) - targetCor), range(cor(rst) - targetCor)))
b = seq(b[1], b[2], len = 100)
hist(cor(result$X) - targetCor, col = scales::alpha("darkblue", 0.3), border = NA, breaks = b); hist(cor(rst) - targetCor, breaks = 100, add = T, col = scales::alpha("red", 0.3), border = NA)
sum((cor(result$X) - targetCor) ^ 2)
sum((cor(rst) - targetCor) ^ 2)




optX = result$X + 0.0
optCor = cor(result$X)
postSimOpt(optX, optCor, targetCor, seed = 123, convergenceTail = 10000)
# range(cor(result$X) - targetCor);
range(cor(optX) - targetCor)
sum((optCor - targetCor) ^ 2)
# range(cor(result$X) / targetCor-1);
range(cor(optX) / targetCor-1)




tmp = marginals + 0.0
tmpCor = cor(tmp)
# postSimOpt(tmp, tmpCor, targetCor, seed = 123, convergenceTail = 10000)
postSimOpt1(tmp, tmpCor, targetCor, seed = 123, convergenceTail = 10000)
range(tmpCor - targetCor)
sum((tmpCor - targetCor) ^ 2)
range(tmpCor / targetCor - 1)


tmp = marginals + 0.0
tmpCor = cor(tmp)
# postSimOpt(tmp, tmpCor, targetCor, seed = 123, convergenceTail = 10000)
postSimOpt2(tmp, tmpCor, targetCor, seed = 123, convergenceTail = 10000)
range(tmpCor - targetCor)
sum((tmpCor - targetCor) ^ 2)
range(tmpCor / targetCor - 1)


tmp = marginals + 0.0
tmpCor = cor(tmp)
postSimOpt(tmp, tmpCor, targetCor, seed = 123, convergenceTail = 10000, acceptProb = c(1))
# postSimOpt1(tmp, tmpCor, targetCor, seed = 123, convergenceTail = 10000)
range(tmpCor - targetCor)
sum((tmpCor - targetCor) ^ 2)
range(tmpCor / targetCor - 1)


tmp = SJMC::postSimOpt(result$X, cor = targetCor, acceptProb = 1, seed = 123, convergenceTail = 10000)


tmp = SJMC::postSimOpt(marginals, targetCor, seed = 123, convergenceTail = 10000, acceptProb = 1)







































