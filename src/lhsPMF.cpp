# include <RcppArmadillo.h>
# include "hpp/LHSsorted.hpp"
using namespace Rcpp;
# define RNG std::mt19937_64


// [[Rcpp::export]]
NumericVector LHSpmf(List pmf, int sampleSize, IntegerVector seed)
{
  RNG rng;
  RNG *rngPtr = &rng;
  if(seed.size() == 1) rng.seed(seed[0]);
  else rngPtr = (RNG*)&seed[0];
  NumericVector val = pmf[0], p = pmf[1];
  NumericVector rst(sampleSize);
  LHSsorted<int, double, double, double, RNG> (
      &rst[0], sampleSize, &val[0], &p[0], val.size(), *rngPtr);
  std::shuffle(rst.begin(), rst.end(), *rngPtr);
  return rst;
}


// [[Rcpp::export]]
IntegerVector exportRandomState(int seed)
{
  RNG rng(seed);
  IntegerVector rst(sizeof(rng) / sizeof(int));
  std::memcpy(&rst[0], &rng, sizeof(rng));
  return rst;
}





















