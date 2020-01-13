# pragma once
# include <Rcpp.h>
using namespace Rcpp;


template<typename rengine>
inline void seedrng(IntegerVector seed, rengine *rng)
{
  if(seed.size() < 4) { rng->seed(seed[0]); return; }
  std::size_t bits[2];
  std::memcpy(bits, &seed[0], sizeof(std::size_t) * 2);
  rng->seed(bits[0]);
  rng->advance(bits[1]);
}


template<typename rengine>
inline void rngseed(rengine *rng, IntegerVector seed)
{
  if(seed.size() < 4) return;
  std::size_t bits[2];
  std::memcpy(&bits[0], &seed[0], sizeof(std::size_t));
  rengine originRng(bits[0]);
  bits[1] = std::size_t(*rng - originRng);
  std::memcpy(&seed[0], bits, sizeof(std::size_t) * 2);
}










