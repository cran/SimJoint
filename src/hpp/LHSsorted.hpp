# pragma once


template<typename indtype, typename sampletype,
         typename losstype, typename probtype,
         typename rengine>
inline void LHSsorted(
    sampletype *sample, indtype sampleSize,
    losstype *x, probtype *p, indtype distSize, rengine &rng)
{
  // Use double regardless of probtype and losstype to gain accuracy.
  double interval = 1.0 / sampleSize;
  std::uniform_real_distribution<double> U(0.0, interval); // Trivial.
  double cumP = p[0];
  indtype j = 0;
  for(indtype i = 0; i < sampleSize; ++i)
  {
    double r = U(rng) + i * interval;
    while(j < distSize and r > cumP)
    {
      ++j;
      cumP += p[j];
    }
    sample[i] = x[j];
  }
}


