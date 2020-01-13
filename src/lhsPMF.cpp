// [[Rcpp::plugins(cpp11)]]
# include <RcppArmadillo.h>
# include "hpp/LHSsorted.hpp"
# include "pcg/pcg_random.hpp"
# include "pcg/toSeed.hpp"
using namespace Rcpp;
// # define RNG std::mt19937_64
# define RNG pcg64


// [[Rcpp::export]]
NumericVector LHSpmf(List pmf, int sampleSize, IntegerVector seed)
{
  RNG rng; seedrng(seed, &rng);
  NumericVector val = pmf[0], p = pmf[1];
  NumericVector rst(sampleSize);
  LHSsorted<int, double, double, double, RNG> (
      &rst[0], sampleSize, &val[0], &p[0], val.size(), rng);
  std::shuffle(rst.begin(), rst.end(), rng);
  rngseed(&rng, seed);
  return rst;
}


// [[Rcpp::export]]
IntegerVector exportRandomState(IntegerVector seed)
{
  if(seed.size() >= 4) return seed;
  RNG rng(seed[0]);
  IntegerVector rst(4);
  rngseed<RNG> (&rng, rst);
  return rst;
}




















































