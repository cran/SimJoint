# pragma once
// [[Rcpp::plugins("cpp11")]]
# include <Rcpp.h>
# include "pcg_random.hpp"
using namespace Rcpp;


// [[Rcpp::export]]
List parallelSeeds(int seed, int streamN)
{
  List rst(streamN);
  pcg64 rengine(seed);
  int n = sizeof(pcg64) / sizeof(int);
  std::size_t step = -1; // = 2 ^ 64 - 1
  for(int i = 0; i < streamN; ++i)
  {
    IntegerVector tmp(n);
    std::memcpy(&tmp[0], &rengine, sizeof(rengine));
    rst[i] = tmp;
    rengine.advance(step);
  }
  return rst;
}


pcg64 getRng(IntegerVector state)
{
  if(state.size() == 1) return pcg64(state[0]);
  pcg64 rst(0);
  std::memcpy(&rst, &state[0], sizeof(int) * state.size());
  return rst;
}






































