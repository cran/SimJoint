// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppParallel)]]
# define ARMA_DONT_PRINT_ERRORS
// [[Rcpp::depends(RcppArmadillo)]]
# include <RcppArmadillo.h>
# include <RcppParallel.h>
# include "hpp/dnyTasking.hpp"
# include "hpp/LHSsorted.hpp"
using namespace Rcpp;
# define vec std::vector
# define INT int64_t
# define RNG std::mt19937_64
# define rptr_ *__restrict__


template<typename indtype, typename valtype>
struct compare
{
  valtype *a;
  compare(valtype *a): a(a){}
  bool operator() (indtype i, indtype j) { return a[i] < a[j]; }
};


// x is a sorted array and y is an unsorted array of the same size.
// This function reorders x such that xReordered and y are
// perfectly rank-correlated. y and xReordered can be the same
// container.
template<typename indtype, typename valtype0, typename valtype1>
void reorder(valtype0 *x, indtype xsize, valtype1 *y,
             valtype0 *xReordered, vec<indtype> &recycle)
{
  recycle.resize(xsize);
  for (indtype i = 0; i < xsize; ++i) recycle[i] = i;
  std::sort(recycle.begin(), recycle.end(),
            compare<indtype, valtype1> (y));
  for (indtype i = 0; i < xsize; ++i) xReordered[recycle[i]] = x[i];
}


// Populate Y[, i] with elements in X[, i] while preserving
// ranks of elements in the original Y[, i].
template<typename indtype, typename valtype0, typename valtype1>
struct rankCorrelate: public RcppParallel::Worker
{
  indtype N;
  valtype0 *X; // N x K matrix. Column vectors are sorted.
  valtype1 *Y; // N x K matrix. Column vectors are unsorted.
  vec<indtype> *auxVec; // Size equals the number of threads.
  dynamicTasking *dT;
  void operator() (std::size_t st, std::size_t end)
  {
    for(;;)
    {
      std::size_t objI = 0;
      if(!dT->nextTaskID(objI)) break;
      INT offset = objI * N;
      reorder<indtype, valtype0, valtype1> (
          X + offset, N, Y + offset, Y + offset, auxVec[st]);
    }
  }
  rankCorrelate(indtype N, indtype K, valtype0 *X, valtype1 *Y,
                vec<vec<indtype> > &auxVectors, int maxCore):
    N(N), X(X), Y(Y)
  {
    dynamicTasking dt(maxCore, K); dT = &dt;
    auxVec = &auxVectors[0];
    parallelFor(0, maxCore, *this);
  }
};




/*
// [[Rcpp::export]]
NumericVector reorderTest(NumericVector x, NumericVector y)
{
  NumericVector rst(x.size());
  vec<int> rc;
  reorder<int, double> (
      &*x.begin(), x.size(), &*y.begin(), &*rst.begin(), rc);
  return rst;
}
*/




template<typename indtype, typename valtype0, typename valtype1, typename valtype2>
struct paraMatMulTriMat: public RcppParallel::Worker
{
  indtype N, K;
  valtype0 *X; // Full matrix on the left. N x K, row-major.
  valtype1 *R; // Upper triangle matrix on the right. K x K.
  valtype2 *rst; // N x K, column-major. Compute rst = X * R.
  dynamicTasking *dT;
  void operator() (std::size_t st, std::size_t end)
  {
    for(;;)
    {
      std::size_t objI = 0;
      if(!dT->nextTaskID(objI)) break;
      objI = K - 1 - objI;
      std::size_t siz = objI + 1;
      valtype1 *Rstart = R + objI * K;
      valtype2 *rstCol = rst + objI * N;
      for(indtype i = 0; i < N; ++i)
        rstCol[i] = std::inner_product(Rstart, Rstart + siz, X + i * K, 0.0);
    }
  }
  paraMatMulTriMat(indtype N, indtype K, valtype0 *X,
                   valtype1 *R, valtype2 *rst, int maxCore):
    N(N), K(K), X(X), R(R), rst(rst)
  {
    dynamicTasking dt(maxCore, K); dT = &dt;
    parallelFor(0, maxCore, *this);
  }
};




template<typename indtype, typename valtype0, typename valtype1, typename valtype2>
struct paraMatMulFullMat: public RcppParallel::Worker
{
  indtype N, P, K;
  valtype0 *X; // Full matrix on the left. N x P, row-major.
  valtype1 *R; // Upper triangle matrix on the right. P x K.
  valtype2 *rst; // N x K, column-major. Compute rst = X * R.
  dynamicTasking *dT;
  void operator() (std::size_t st, std::size_t end)
  {
    for(;;)
    {
      std::size_t objI = 0;
      if(!dT->nextTaskID(objI)) break;
      valtype2 *rstCol = rst + objI * N;
      valtype1 *Rcol = R + objI * P;
      for(indtype i = 0; i < N; ++i)
      {
        valtype0 *start = X + i * P;
        rstCol[i] = std::inner_product(start, start + P, Rcol, 0.0);
      }
    }
  }
  paraMatMulFullMat(indtype N, indtype P, indtype K, valtype0 *X,
                    valtype1 *R, valtype2 *rst, int maxCore):
    N(N), P(P), K(K), X(X), R(R), rst(rst)
  {
    dynamicTasking dt(maxCore, K); dT = &dt;
    parallelFor(0, maxCore, *this);
  }
};




template<typename indtype, typename sampletype, typename cortype>
struct correlation: public RcppParallel::Worker
{
  indtype N, K;
  sampletype *X;
  cortype *C; // C is a lower triangle matrix.
  dynamicTasking *dT;
  void operator() (std::size_t st, std::size_t end)
  {
    for(;;)
    {
      std::size_t objI = 0;
      if(!dT->nextTaskID(objI)) break;
      cortype *rst = C + objI * (K - 1 + K - objI) / 2;
      for(indtype i = 0, iend = K - 1 - objI; i < iend; ++i)
      {
        indtype theOtherCol = objI + i + 1;
        rst[i] = std::inner_product(
          X + objI * N, X + objI * N + N, X + theOtherCol * N, 0.0);
      }
    }
  }
  correlation(indtype N, indtype K, sampletype *X, cortype *C, int maxCore):
    N(N), K(K), X(X), C(C)
  {
    dynamicTasking dt(maxCore, K - 1); dT = &dt;
    parallelFor(0, maxCore, *this);
  }
};




// errType == 0: average absolute relative error.
// errType == 1: max absolute relative error.
// errType == 2: mean square
template<typename indtype, typename valtype, int errType>
valtype overallErr(valtype *targetFullCorMat,
                   indtype K, valtype *lowTriCorMat)
{
  indtype t = 0;
  valtype err = 0;


  if(errType == 0) // Average absolute relative error.
  {
    for(indtype i = 0; i < K; ++i)
    {
      valtype *x = targetFullCorMat + K * i;
      for(indtype j = i + 1; j < K; ++j)
      {
        double tmpErr;
        if(x[j] == 0) tmpErr = std::abs(lowTriCorMat[t] - x[j]);
        else tmpErr = std::abs(lowTriCorMat[t] / x[j] - 1);
        err += tmpErr;
        ++t;
      }
    }
    err /= (K * (K - 1) / 2);
  }


  else if(errType == 1) // Max absolute relative error
  {
    for(indtype i = 0; i < K; ++i)
    {
      valtype *x = targetFullCorMat + K * i;
      for(indtype j = i + 1; j < K; ++j)
      {
        double tmpErr;
        if(x[j] == 0) tmpErr = std::abs(lowTriCorMat[t] - x[j]);
        else tmpErr = std::abs(lowTriCorMat[t] / x[j] - 1);
        if(tmpErr > err) err = tmpErr;
        ++t;
      }
    }
  }


  else if(errType == 2) // mean square
  {
    for(indtype i = 0; i < K; ++i)
    {
      valtype *x = targetFullCorMat + K * i;
      for(indtype j = i + 1; j < K; ++j)
      {
        double tmpErr = lowTriCorMat[t] - x[j];
        err += tmpErr * tmpErr;
        ++t;
      }
    }
    err = std::sqrt(err / (K * (K - 1) / 2));
  }


  return err;
}




template<typename indtype, typename valtype>
void copyLowerTriMat(valtype *X, indtype K, valtype *theCopy)
{
  indtype t = 0;
  for(indtype i = 0; i < K; ++i)
  {
    valtype *x = X + i * K;
    for(indtype j = i + 1; j < K; ++j)
    {
      theCopy[t] = x[j];
      ++t;
    }
  }
}




template<typename indtype, typename valtype>
void copyLowerTriMatToFull(valtype *X, indtype K,
                           valtype *triMat, valtype diag = 1)
{
  indtype t = 0;
  for(indtype i = 0; i < K; ++i)
  {
    valtype *x = X + i * K;
    x[i] = diag;
    for(indtype j = i + 1; j < K; ++j)
    {
      x[j] = triMat[t];
      ++t;
    }
  }
}




// Adjust the lower triangle of correlation matrix.
template<typename indtype, typename valtype>
void adjustCorMat(
    indtype K,
    valtype rptr_ targetCorMat,
    valtype rptr_ bestTriCorMatReached,
    valtype rptr_ stepRatio,
    valtype rptr_ initialDummyTargetTriCorMat,
    valtype rptr_ dummyTargetCorMat)
{
  indtype t = 0;
  for(indtype i = 0; i < K; ++i)
  {
    valtype rptr_ targetCorMatX = targetCorMat + i * K;
    valtype rptr_ dummyTargetCorMatX = dummyTargetCorMat + i * K;
    for(indtype j = i + 1; j < K; ++j)
    {
      dummyTargetCorMatX[j] = std::max<valtype> (
        1e-5 - 1, std::min<valtype> (
            1 - 1e-5, initialDummyTargetTriCorMat[t] +
              (targetCorMatX[j] - bestTriCorMatReached[t]) * stepRatio[t]));
      ++t;
    }
  }
}




// Reflect the lower triangle matrix to the upper triangle.
template<typename valtype>
void reflectLowerTri(arma::Mat<valtype> &squareMat)
{
  INT dim = squareMat.n_cols;
  for(INT i = 1; i < dim; ++i)
  {
    for(INT j = 0; j < i; ++j)
      squareMat.at(j, i) = squareMat.at(i, j);
  }
}




// Triangle matrix represented as a flat vector.
// Convert it to a full matrix.
template<typename valtype>
void triMat2full(valtype *tri, arma::Mat<valtype> &fullMat)
{
  INT K = fullMat.n_cols;
  INT t = 0;
  for(INT i = 0; i < K; ++i)
  {
    valtype *x = &fullMat[0] + i * K;
    x[i] = 1;
    for(INT j = i + 1; j < K; ++j)
    {
      x[j] = tri[t];
      ++t;
    }
  }
  reflectLowerTri(fullMat);
}




// Return 0: factorization failed.
// Return 1: Use triangle matrix multiplication against the result.
// Return 2: Use full matrix multiplication against the result.
template<bool verbose, typename valtype>
inline int factorize(
    arma::Mat<valtype> &C, arma::Mat<valtype> &rst, arma::Col<valtype> &ev)
{
  bool egSuccess = true;
  bool cholSuccess = arma::chol(rst, C);
  if(cholSuccess) return 1;
  if(verbose) Rcout << "Cholesky decomposition failed. Perform eigen decomposition.\n";
  egSuccess = arma::eig_sym(ev, rst, C);
  if(!egSuccess)
  {
    if(verbose) Rcout << "Invalid correlation matrix.\n";
    return 0;
  }
  // Avoid numeric crap like -1e-17 as an eigen value:
  for(unsigned i = 0, iend = ev.size(); i < iend; ++i)
    ev[i] = std::sqrt(std::max<double> (ev[i], 0));
  arma::inplace_trans(rst);
  for(unsigned i = 0, iend = rst.n_cols; i < iend; ++i)
    for(unsigned j = 0, jend = rst.n_rows; j < jend; ++j)
      rst.at(j, i) *= ev[j];
  return 2;
}




// errType == 0: mean relative error.
// errType == 1: max relative error.
// errType == 2: mean square.
// X has been shift and scaled such that cross products directly equal
// to correlations.
template <bool useSupportX, bool verbose, int errType,
          typename sampletype, typename cortype>
void simJointPearsonTemplate(
    arma::Mat<sampletype> &X,
    arma::Mat<cortype> &cor,
    arma::Mat<sampletype> &supportX,
    RNG &rng,
    arma::Mat<sampletype> &reorderedX, // Result.
    arma::Mat<cortype> &finalCor, // Correlation matrix of the result.
    NumericVector stochasticStepDomain = NumericVector::create(0, 1),
    int maxCore = 7,
    int convergenceTail = 8,
    int iterLimit = 100000)
{
  maxCore = std::min<int> (maxCore, cor.n_cols);


  INT N = X.n_rows, K = X.n_cols;
  arma::Mat<sampletype> Y(N, K);
  arma::Mat<sampletype> XrContainer;
  arma::Mat<sampletype> *Xr = nullptr;


  if(!useSupportX)
  {
    std::copy(X.begin(), X.end(), Y.begin());
    //==========================================================================
    for(INT i = 0; i < K; ++i)
    {
      sampletype *y = &Y[0] + N * i;
      std::shuffle(y, y + N, rng);
    }
    XrContainer = Y.t(); // Xr is row-major column-randomized X.
    // Xr is used to left-multiply the upper triangle of Cholesky decomposition,
    // and the result is stored in Y.
    // ==========================================================================
    Xr = &XrContainer;
  }
  else Xr = &supportX;


  arma::Mat<cortype> &targetCorMat = cor;
  arma::Mat<cortype> factMat(K, K);
  arma::Col<cortype> eigenVal(K);
  arma::Mat<cortype> dummyTargetCorMat = cor;
  vec<cortype> initialDummyTargetTriCorMat(K * (K - 1) / 2);
  copyLowerTriMat(&dummyTargetCorMat[0], K, &initialDummyTargetTriCorMat[0]);
  vec<cortype> stepRatio(initialDummyTargetTriCorMat.size());


  // The lower triangle of the correlation matrix.
  vec<cortype> corTriMatReached(initialDummyTargetTriCorMat.size());
  vec<cortype> bestCorTriMatReached(corTriMatReached.size());
  // The above containers are recyclable.


  const int Nrecord = convergenceTail;
  vec<double> errorRecords(Nrecord);
  double *errRecords = &errorRecords[0];
  {
    std::uniform_real_distribution<double> U(
        std::numeric_limits<double>::max() / 2,
        std::numeric_limits<double>::max());
    for(int i = 0; i < Nrecord; ++i) errRecords[i] = U(rng);
  }


  // eigenVal might not be used in this function.
  int fact = factorize<verbose, cortype> (dummyTargetCorMat, factMat, eigenVal);
  if(fact == 0)
  {
    Rcout << "Quit.\n";
    return;
  }


  vec<vec<uint_fast32_t> > auxForImanConover(maxCore, vec<uint_fast32_t> (N));
  double bestErr = std::numeric_limits<double>::max();


  // Iterative correlation algorithm begins.
  for(int iter = 0; iter < iterLimit; ++iter)
  {
    if(verbose) Rcout << "Iteration = " << iter + 1 << ": ";


    if(fact == 1) paraMatMulTriMat<INT, sampletype, cortype, sampletype> (
      N, K, &(*Xr)[0], &factMat[0], &Y[0], maxCore);
    else paraMatMulFullMat<INT, sampletype, cortype, sampletype> (
        N, K, K, &(*Xr)[0], &factMat[0], &Y[0], maxCore);


    rankCorrelate<uint_fast32_t, sampletype, sampletype> (
        N, K, &X[0], &Y[0], auxForImanConover, maxCore);


    // New correlation matrix is in corMatReached.
    correlation<INT, sampletype, cortype> (
        N, K, &Y[0], &corTriMatReached[0], maxCore);


    double err = overallErr<INT, double, errType> (
      &targetCorMat[0], K, &corTriMatReached[0]);


    if(verbose)
    {
      if(errType == 0) Rcout << "mean abs relative error in cor = ";
      else if(errType == 1) Rcout << "max abs relative error in cor = ";
      else Rcout << "square root of mean squared error in cor = ";
      Rcout << err << "\n";
    }


    std::copy(errRecords + 1, errRecords + Nrecord, errRecords);
    errRecords[Nrecord - 1] = err;


    if(true) // Should I break the loop.
    {
      bool shouldBreak = true;
      for(int i = 0; i < Nrecord; ++i)
      {
        if(std::abs(errRecords[Nrecord - 1] / errRecords[i] - 1) < 1e-5)
          continue;
        shouldBreak = false;
        break;
      }
      if(shouldBreak) break;
    }


    double lowUnif = stochasticStepDomain[0];
    double higUnif = stochasticStepDomain[1];
    while(true)
    {
      if(err < bestErr)
      {
        bestCorTriMatReached.swap(corTriMatReached);
        bestErr = err;
        std::fill(stepRatio.begin(), stepRatio.end(), 1.0);
        copyLowerTriMat(&dummyTargetCorMat[0], K, &initialDummyTargetTriCorMat[0]);
      }
      else
      {
        if(lowUnif == higUnif)
        {
          for(INT i = 0, iend = stepRatio.size(); i < iend; ++i)
            stepRatio[i] *= higUnif;
        }
        else
        {
          std::uniform_real_distribution<double> U(lowUnif, higUnif);
          // std::uniform_int_distribution<signed char> intU(0, 1);
          for(INT i = 0, iend = stepRatio.size(); i < iend; ++i)
            stepRatio[i] *= U(rng);
            // stepRatio[i] *= U(rng) * intU(rng);
        }
      }


      // Adjust dummyTargetCorMat.
      adjustCorMat<INT, double> (
          K, &targetCorMat[0], &bestCorTriMatReached[0], &stepRatio[0],
          &initialDummyTargetTriCorMat[0], &dummyTargetCorMat[0]);


      reflectLowerTri(dummyTargetCorMat);


      // Return 0: factorization failed.
      // Return 1: Use triangle matrix multiplication against the result.
      // Return 2: Use full matrix multiplication against the result.
      fact = factorize<verbose, cortype> (dummyTargetCorMat, factMat, eigenVal);
      if(fact != 0) break;
      if(verbose) Rcout << "Simulate new correlation matrix.\n";


      lowUnif = -1;
      higUnif = 1;
    }


  }


  triMat2full<cortype> (&corTriMatReached[0], dummyTargetCorMat);
  reorderedX.swap(Y);
  finalCor.swap(dummyTargetCorMat);
}




template<typename valtype>
void normalize(valtype *X, INT N, INT K,
               vec<double> &shift, vec<double> &multiplier)
{
  shift.resize(K);
  multiplier.resize(K);
  double sqrtN = std::sqrt(N + 0.0);
  for(INT i = 0; i < K; ++i)
  {
    valtype *x = X + i * N;
    double m = 0, ss = 0;
    for(INT j = 0; j < N; ++j)
    {
      double tmp = x[j];
      m += tmp;
      ss += tmp * tmp;
    }
    m /= N;
    double mul = 1.0 / (std::sqrt(ss / N - m * m) * sqrtN);
    shift[i] = m;
    multiplier[i] = mul;
    for(INT j = 0; j < N; ++j)
      x[j] = (x[j] - m) * mul;
  }
}


template<typename valtype>
void recover(valtype *X, INT N, INT K,
             vec<double> &shift, vec<double> &multiplier)
{
  for(INT i = 0; i < K; ++i)
  {
    valtype *x = X + i * N;
    double m = shift[i], mul = 1.0 / multiplier[i];
    for(INT j = 0; j < N; ++j)
    {
      x[j] = x[j] * mul + m;
    }
  }
}




// X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
// seed, maxCore, convergenceTail, iterLimit

template <bool pearson, bool useSupportX, bool usePMF,
          bool verbose, int errType, typename sampletype,
          typename cortype>
List simJointTemplate(
    arma::Mat<sampletype> &X_,
    List PMFs,
    int sampleSize,
    arma::Mat<cortype> &cor,
    arma::Mat<sampletype> &supportX,
    NumericVector stochasticStepDomain = NumericVector::create(0, 1),
    IntegerVector seed = 123,
    int maxCore = 7,
    int convergenceTail = 8,
    int iterLimit = 100000)
{


  if(true) // Check integrity.
  {
    if(X_.size() > 0)
    {
      if(X_.n_cols != cor.n_cols)
      {
        Rcerr << "`X` and `cor` do not have the same dimensionality (columns). Quit.\n";
        return List::create();
      }
      INT N = X_.n_rows, K = X_.n_cols;
      for(INT i = 0; i < K; ++i)
      {
        sampletype *x = &X_[0] + i * N;
        for(INT j = 1; j < N; ++j)
        {
          if(x[j] >= x[j - 1]) continue;
          Rcerr << "`X` has unsorted columns. Quit.\n";
          return List::create();
        }
      }
    }
    else if(PMFs.size() > 0)
    {
      if(PMFs.size() - cor.n_cols != 0)
      {
        Rcerr << "`PMFs` and `cor` have different numbers of columns. Quit.\n";
        return List::create();
      }
      for(INT i = 0, iend = PMFs.size(); i < iend; ++i)
      {
        List pmf = PMFs[i];
        if(pmf.size() < 2)
        {
          Rcerr << "A PMF has less than 2 columns. Quit.\n";
          return List::create();
        }
        NumericVector x = pmf[0], p = pmf[1];
        if(x.size() <= 1)
        {
          Rcerr << "`PMFs` contains degenerate distributions. Quit.\n";
          return List::create();
        }
        if(x.size() != p.size())
        {
          Rcerr << "Value and probability vectors of a PMF have different sizes. Quit.\n";
          return List::create();
        }
      }
    }
    if(supportX.size() > 0 and supportX.n_cols != cor.n_cols)
    {
      Rcerr << "`supportX` and `cor` do not have the same dimensionality (columns). Quit.\n";
      return List::create();
    }
  }




  arma::Mat<sampletype> *X = &X_;
  arma::Mat<sampletype> Xcontainer;
  RNG rengine, *rng = &rengine;
  if(seed.size() == 1) rng->seed(seed[0]);
  else
  {
    if(seed.size() * sizeof(int) != sizeof(RNG))
    {
      Rcerr << "`seed` should be either an integer or an integer vector of size 626. Quit.\n";
      return List::create();
    }
    rng = (RNG*)&seed[0];
  }


  if(usePMF)
  {
    int N = sampleSize, K = PMFs.size();
    Xcontainer.set_size(N, K);
    X = &Xcontainer;
    for(INT i = 0; i < K; ++i)
    {
      List pmf = PMFs[i];
      NumericVector x = pmf[0], p = pmf[1];
      LHSsorted<int, sampletype, double, double, RNG> (
          X->begin() + i * N, N, &x[0], &p[0], x.size(), *rng);
    }
  }


  INT N = (*X).n_rows, K = (*X).n_cols;
  arma::Mat<float> Xrank;
  if(!pearson)
  {
    Xrank.set_size(N, K);
    for(INT i = 0; i < K; ++i)
    {
      INT offset = i * N;
      double *x = &(*X)[0] + offset;
      float *xr = &Xrank[0] + offset;
      for(INT j = 0; j < N; )
      {
        INT k = j + 1;
        while(k < N and x[k] <= x[j]) ++k;
        double rk = (k - 1 + j) / 2.0;
        for(; j < k; ++j) xr[j] = rk;
      }
    }
  }


  // Normalization
  vec<double> shift, multiplier;
  if(pearson) normalize<sampletype> (&(*X)[0], N, K, shift, multiplier);
  else normalize<float> (&Xrank[0], N, K, shift, multiplier);


  arma::Mat<sampletype> reorderedX;
  arma::Mat<float> reorderedXfloat;
  arma::Mat<cortype> finalCor;
  if(!pearson)
  {
    arma::Mat<float> tmpSupportX;
    simJointPearsonTemplate<false, verbose, errType, float, cortype> (
        Xrank, cor, tmpSupportX, *rng, reorderedXfloat, finalCor,
        stochasticStepDomain, maxCore, convergenceTail, iterLimit);
  }
  else
  {
    simJointPearsonTemplate<useSupportX, verbose, errType, sampletype, cortype> (
        *X, cor, supportX, *rng, reorderedX, finalCor, stochasticStepDomain,
        maxCore, convergenceTail, iterLimit);
  }


  if(pearson)
  {
    recover<sampletype> (&reorderedX[0], N, K, shift, multiplier);
    if(!usePMF and !useSupportX) recover<sampletype> (
        &(*X)[0], N, K, shift, multiplier);
  }
  else
  {
    recover<float> (&reorderedXfloat[0], N, K, shift, multiplier);
    reorderedX.set_size(N, K);
    for(INT i = 0; i < K; ++i)
    {
      INT offset = i * N;
      float *xr = &reorderedXfloat[0] + offset;
      sampletype *x = &(*X)[0] + offset;
      sampletype *xrdr = &reorderedX[0] + offset;
      for(INT j = 0; j < N; ++j)
          xrdr[j] = x[INT(std::round(xr[j]))];
    }
  }


  return List::create(Named("X") = reorderedX, Named("cor") = finalCor);
}












// ============================================================================
// [[Rcpp::export]]
List SJpearson(
    arma::mat &X,
    arma::mat &cor,
    NumericVector stochasticStepDomain = NumericVector::create(0, 1),
    Rcpp::String errorType = "meanSquare", // "maxRela", "meanSquare"
    IntegerVector seed = 123,
    int maxCore = 7,
    int convergenceTail = 8,
    int iterLimit = 100000,
    bool verbose = true
)
{
  arma::mat noise;
  List PMFs;
  int sampleSize = 0;
  // <bool pearson, bool useSupportX, bool usePMF,
  //  bool verbose, int errType, typename sampletype,
  //  typename cortype>
  List rst;
  int branch = 0;
  if(errorType == "meanRela") branch = int(verbose) * 10 + 0;
  else if(errorType == "maxRela") branch = int(verbose) * 10 + 1;
  else if(errorType == "meanSquare") branch = int(verbose) * 10 + 2;
  if(branch == 00) rst = simJointTemplate     <true, false, false, 0, 0, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  else if(branch == 01) rst = simJointTemplate<true, false, false, 0, 1, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  else if(branch == 02) rst = simJointTemplate<true, false, false, 0, 2, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  else if(branch == 10) rst = simJointTemplate<true, false, false, 1, 0, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  else if(branch == 11) rst = simJointTemplate<true, false, false, 1, 1, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  else rst = simJointTemplate                 <true, false, false, 1, 2, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  return rst;
}








// [[Rcpp::export]]
List SJpearsonPMF(
    Rcpp::List PMFs,
    int sampleSize,
    arma::mat &cor,
    NumericVector stochasticStepDomain = NumericVector::create(0, 1),
    Rcpp::String errorType = "meanSquare", // "maxRela", "meanSquare"
    IntegerVector seed = 123,
    int maxCore = 7,
    int convergenceTail = 8,
    int iterLimit = 100000,
    bool verbose = true
)
{
  arma::mat noise;
  arma::mat X;
  // <bool pearson, bool useSupportX, bool usePMF, bool verbose, int errType>
  List rst;
  int branch = 0;
  if(errorType == "meanRela") branch = int(verbose) * 10 + 0;
  else if(errorType == "maxRela") branch = int(verbose) * 10 + 1;
  else if(errorType == "meanSquare") branch = int(verbose) * 10 + 2;
  if(branch == 00) rst = simJointTemplate     <true, false, true, 0, 0, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  else if(branch == 01) rst = simJointTemplate<true, false, true, 0, 1, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  else if(branch == 02) rst = simJointTemplate<true, false, true, 0, 2, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  else if(branch == 10) rst = simJointTemplate<true, false, true, 1, 0, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  else if(branch == 11) rst = simJointTemplate<true, false, true, 1, 1, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  else rst = simJointTemplate                 <true, false, true, 1, 2, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  return rst;
}




// [[Rcpp::export]]
List xSJpearson(
    arma::mat &X,
    arma::mat &cor,
    arma::mat &noise,
    NumericVector stochasticStepDomain = NumericVector::create(0, 1),
    Rcpp::String errorType = "meanSquare", // "maxRela", "meanSquare"
    IntegerVector seed = 123,
    int maxCore = 7,
    int convergenceTail = 8,
    int iterLimit = 100000,
    bool verbose = true
)
{
  List PMFs;
  int sampleSize = 0;
  // <bool pearson, bool useSupportX, bool usePMF, bool verbose, int errType>
  List rst;
  int branch = 0;
  if(errorType == "meanRela") branch = int(verbose) * 10 + 0;
  else if(errorType == "maxRela") branch = int(verbose) * 10 + 1;
  else if(errorType == "meanSquare") branch = int(verbose) * 10 + 2;
  if(branch == 00) rst = simJointTemplate     <true, true, false, 0, 0, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  else if(branch == 01) rst = simJointTemplate<true, true, false, 0, 1, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  else if(branch == 02) rst = simJointTemplate<true, true, false, 0, 2, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  else if(branch == 10) rst = simJointTemplate<true, true, false, 1, 0, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  else if(branch == 11) rst = simJointTemplate<true, true, false, 1, 1, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  else rst = simJointTemplate                 <true, true, false, 1, 2, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  return rst;
}




// [[Rcpp::export]]
List xSJpearsonPMF(
    Rcpp::List PMFs,
    int sampleSize,
    arma::mat &cor,
    arma::mat &noise,
    NumericVector stochasticStepDomain = NumericVector::create(0, 1),
    Rcpp::String errorType = "meanSquare", // "maxRela", "meanSquare"
    IntegerVector seed = 123,
    int maxCore = 7,
    int convergenceTail = 8,
    int iterLimit = 100000,
    bool verbose = true
)
{
  arma::mat X;
  // <bool pearson, bool useSupportX, bool usePMF, bool verbose, int errType>
  List rst;
  int branch = 0;
  if(errorType == "meanRela") branch = int(verbose) * 10 + 0;
  else if(errorType == "maxRela") branch = int(verbose) * 10 + 1;
  else if(errorType == "meanSquare") branch = int(verbose) * 10 + 2;
  if(branch == 00) rst = simJointTemplate     <true, true, true, 0, 0, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  else if(branch == 01) rst = simJointTemplate<true, true, true, 0, 1, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  else if(branch == 02) rst = simJointTemplate<true, true, true, 0, 2, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  else if(branch == 10) rst = simJointTemplate<true, true, true, 1, 0, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  else if(branch == 11) rst = simJointTemplate<true, true, true, 1, 1, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  else rst = simJointTemplate                 <true, true, true, 1, 2, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  return rst;
}




// ============================================================================
// [[Rcpp::export]]
List SJspearman(
    arma::mat &X,
    arma::mat &cor,
    NumericVector stochasticStepDomain = NumericVector::create(0, 1),
    Rcpp::String errorType = "meanSquare", // "maxRela", "meanSquare"
    IntegerVector seed = 123,
    int maxCore = 7,
    int convergenceTail = 8,
    int iterLimit = 100000,
    bool verbose = true
)
{
  arma::mat noise;
  List PMFs;
  int sampleSize = 0;
  // <bool pearson, bool useSupportX, bool usePMF, bool verbose, int errType>
  List rst;
  int branch = 0;
  if(errorType == "meanRela") branch = int(verbose) * 10 + 0;
  else if(errorType == "maxRela") branch = int(verbose) * 10 + 1;
  else if(errorType == "meanSquare") branch = int(verbose) * 10 + 2;
  if(branch == 00) rst = simJointTemplate     <false, false, false, 0, 0, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  else if(branch == 01) rst = simJointTemplate<false, false, false, 0, 1, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  else if(branch == 02) rst = simJointTemplate<false, false, false, 0, 2, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  else if(branch == 10) rst = simJointTemplate<false, false, false, 1, 0, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  else if(branch == 11) rst = simJointTemplate<false, false, false, 1, 1, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  else rst = simJointTemplate                 <false, false, false, 1, 2, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  return rst;
}




// [[Rcpp::export]]
List SJspearmanPMF(
    Rcpp::List PMFs,
    int sampleSize,
    arma::mat &cor,
    NumericVector stochasticStepDomain = NumericVector::create(0, 1),
    Rcpp::String errorType = "meanSquare", // "maxRela", "meanSquare"
    IntegerVector seed = 123,
    int maxCore = 7,
    int convergenceTail = 8,
    int iterLimit = 100000,
    bool verbose = true
)
{
  arma::mat noise;
  arma::mat X;
  // <bool pearson, bool useSupportX, bool usePMF, bool verbose, int errType>
  List rst;
  int branch = 0;
  if(errorType == "meanRela") branch = int(verbose) * 10 + 0;
  else if(errorType == "maxRela") branch = int(verbose) * 10 + 1;
  else if(errorType == "meanSquare") branch = int(verbose) * 10 + 2;
  if(branch == 00) rst = simJointTemplate     <false, false, true, 0, 0, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  else if(branch == 01) rst = simJointTemplate<false, false, true, 0, 1, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  else if(branch == 02) rst = simJointTemplate<false, false, true, 0, 2, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  else if(branch == 10) rst = simJointTemplate<false, false, true, 1, 0, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  else if(branch == 11) rst = simJointTemplate<false, false, true, 1, 1, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  else rst = simJointTemplate                 <false, false, true, 1, 2, double, double> (
    X, PMFs, sampleSize, cor, noise, stochasticStepDomain,
    seed, maxCore, convergenceTail, iterLimit);
  return rst;
}




template<typename valtype>
inline bool sameSign(valtype x, valtype y)
{ return (x < 0) == (y < 0); }


struct mem
{
  double val, *place;
  mem(){}
  mem(double val, double *place): val(val), place(place) {}
};


struct twoCor
{
  double cor, targetCor;
  twoCor(){}
  twoCor(double cor, double targetCor): cor(cor), targetCor(targetCor) {}
};


struct corIndDiff
{
  int I, J;
  double corDiff;
  // bool operator < (const corIndDiff &X) const
  // {
  //   return absCorDiff > X.absCorDiff;
  // }
  corIndDiff(){}
  corIndDiff(int I, int J, double corDiff):
    I(I), J(J), corDiff(corDiff) {}
};


inline void push(int I, int J, double corDiff,
                 corIndDiff *v, int size)
{
  if(std::abs(corDiff) < std::abs(v[0].corDiff)) return;
  v[0].corDiff = corDiff;
  v[0].I = I;
  v[0].J = J;
  for(int i = 1; i < size; ++i)
  {
    if(std::abs(v[i].corDiff) >=
       std::abs(v[i - 1].corDiff)) return;
    std::swap(v[i], v[i - 1]);
  }
}




/*
// [[Rcpp::export]]
void postSimOpt1(arma::mat &X, arma::mat &Xcor, arma::mat &targetCor,
                 IntegerVector seed = 123,
                 int convergenceTail = 10000)
{
  INT N = X.n_rows, K = X.n_cols;
  vec<double> shift, multiplier;
  normalize<double> (&X[0], N, K, shift, multiplier);
  if(Xcor.n_cols - K != 0)
  {
    Rcout << "`Xcor` has wrong number of columns. Quit.\n";
    return;
  }
  if(targetCor.n_cols - K != 0)
  {
    Rcout << "`targetCor` has wrong number of columns. Quit.\n";
    return;
  }


  RNG rengine, *rng = &rengine;
  if(seed.size() == 1) rng->seed(seed[0]);
  else
  {
    if(seed.size() * sizeof(int) != sizeof(RNG))
    {
      Rcerr << "`seed` should be either an integer or an integer vector of size 626. Quit.\n";
      return;
    }
    rng = (RNG*)&seed[0];
  }


  // Find initial worst entries in correlation matrix.
  // Calculate initial cost.
  vec<twoCor> sigma(K * K);
  vec<twoCor*> sigmacol(K);
  for(INT i = 0; i < K; ++i) sigmacol[i] = &sigma[0] + K * i;
  twoCor **Sigma = &sigmacol[0];


  for(INT i = 0, iend = K - 1; i < iend; ++i)
  {
    for(INT j = i + 1; j < K; ++j)
    {
      Sigma[i][j].cor = Xcor.at(j, i);
      Sigma[i][j].targetCor = targetCor.at(j, i);
    }
  }


  INT gap = 0;
  std::uniform_int_distribution<INT> U(0, N - 1);
  std::bernoulli_distribution B(0.5);
  // std::uniform_real_distribution<double> unif(0, 1);
  vec<mem> wouldBeCor(K - 1);
  double *x = &X[0];


  // vec<double> prob(acceptProb.begin(), acceptProb.end());
  // std::reverse(prob.begin(), prob.end());
  // std::partial_sum(prob.begin(), prob.end(), prob.begin());
  // prob.back() = 1;
  // vec<corIndDiff> nnCandidates(prob.size());
  // int nn = nnCandidates.size();


  while(gap < convergenceTail)
  {
    double worstErr = 0;
    INT I = 0, J = 0;
    // for(int i = 0; i < nn; ++i) nnCandidates[i].corDiff = 0;
    for(int i = 0, iend = K - 1; i < iend; ++i)
    {
      for(int j = i + 1, jend = K; j < jend; ++j)
      {
        double tmp = Sigma[i][j].cor - Sigma[i][j].targetCor;
        if(std::abs(tmp) <= std::abs(worstErr)) continue;
        worstErr = tmp;
        I = i;
        J = j;
        // push(i, j, Sigma[i][j].cor - Sigma[i][j].targetCor,
        //      &nnCandidates[0], nn);
      }
    }


    // int chosen = std::lower_bound(
    //   prob.begin(), prob.end(), unif(*rng)) - prob.begin();
    // I = nnCandidates[chosen].I;
    // J = nnCandidates[chosen].J;


    // INT col = I;
    // if(B(rng)) col = J;
    // If we swap x[col][u], x[col][v], what would happen...?
    // First see
    double *P = x + I * N, *Q = x + J * N;
    INT u, v;
    bool negativeWorstErr = worstErr < 0;
    while(true)
    {
      u = U(*rng); v = U(*rng);
      while(u == v) {u = U(*rng); v = U(*rng);}
      // double pdiff = , qdiff = ;
      bool haveSameSign = sameSign(P[u] - P[v], Q[u] - Q[v]);
      // if((negativeWorstErr and !haveSameSign) or
      //    (!negativeWorstErr and haveSameSign)) break;
      if(negativeWorstErr != haveSameSign) break;
    }


    // u, v can reduce the error in the worst entry of correlation matrix,
    // but do they also reduce the cost function value?
    // double *col = I * N + x;
    INT col = I;
    if(B(*rng)) col = J; // swap() elements would apply to this column.
    double costDiff = 0;
    double &colu = X.at(u, col), &colv = X.at(v, col);
    wouldBeCor.resize(0);
    for(INT i = 0; i < K; ++i)
    {
      twoCor *a = nullptr;
      if(i < col) a = &Sigma[i][col];
      else if (i > col) a = &Sigma[col][i];
      else continue;
      double iu = X.at(u, i), iv = X.at(v, i);
      // double currentInnerProd = iu * colu + iv * colv;
      // double tentativeInnerProd = iu * colv + iv * colu;
      // double tentativeCor = a->cor + (tentativeInnerProd - currentInnerProd);
      double tentativeCor = a->cor + (iu - iv) * (colv - colu);
      costDiff += std::pow(tentativeCor - a->targetCor, 2) -
        std::pow(a->cor - a->targetCor, 2);
      wouldBeCor.push_back(mem(tentativeCor, &a->cor));
    }


    if(costDiff < 0)
    {
      gap = 0;
      std::swap(colu, colv);
      for(INT i = 0, iend = wouldBeCor.size(); i < iend; ++i)
        *wouldBeCor[i].place = wouldBeCor[i].val;
    }
    else ++gap;
  }


  recover<double> (&X[0], N, K, shift, multiplier);
  for(INT i = 0, iend = K - 1; i < iend; ++i)
  {
    for(INT j = i + 1; j < K; ++j)
    {
      Xcor.at(j, i) = Sigma[i][j].cor;
      Xcor.at(i, j) = Sigma[i][j].cor;
    }
  }
}




// [[Rcpp::export]]
void postSimOpt2(arma::mat &X, arma::mat &Xcor, arma::mat &targetCor,
                 IntegerVector seed = 123,
                 int convergenceTail = 10000)
{
  INT N = X.n_rows, K = X.n_cols;
  vec<double> shift, multiplier;
  normalize<double> (&X[0], N, K, shift, multiplier);
  if(Xcor.n_cols - K != 0)
  {
    Rcout << "`Xcor` has wrong number of columns. Quit.\n";
    return;
  }
  if(targetCor.n_cols - K != 0)
  {
    Rcout << "`targetCor` has wrong number of columns. Quit.\n";
    return;
  }


  RNG rengine, *rng = &rengine;
  if(seed.size() == 1) rng->seed(seed[0]);
  else
  {
    if(seed.size() * sizeof(int) != sizeof(RNG))
    {
      Rcerr << "`seed` should be either an integer or an integer vector of size 626. Quit.\n";
      return;
    }
    rng = (RNG*)&seed[0];
  }


  // Find initial worst entries in correlation matrix.
  // Calculate initial cost.
  vec<twoCor> sigma(K * K);
  vec<twoCor*> sigmacol(K);
  for(INT i = 0; i < K; ++i) sigmacol[i] = &sigma[0] + K * i;
  twoCor **Sigma = &sigmacol[0];


  for(INT i = 0, iend = K - 1; i < iend; ++i)
  {
    for(INT j = i + 1; j < K; ++j)
    {
      Sigma[i][j].cor = Xcor.at(j, i);
      Sigma[i][j].targetCor = targetCor.at(j, i);
    }
  }


  INT gap = 0;
  std::uniform_int_distribution<INT> U(0, N - 1);
  std::uniform_int_distribution<INT> Uk(0, K - 1);
  // std::bernoulli_distribution B(0.5);
  vec<mem> wouldBeCor(K - 1);
  double *x = &X[0];


  while(gap < convergenceTail)
  {

    INT I = 0, J = 0;
    // for(int i = 0, iend = K - 1; i < iend; ++i)
    // {
    //   for(int j = i + 1, jend = K; j < jend; ++j)
    //   {
    //     double tmp = Sigma[i][j].cor - Sigma[i][j].targetCor;
    //     if(std::abs(tmp) <= std::abs(worstErr)) continue;
    //     worstErr = tmp;
    //     I = i;
    //     J = j;
    //   }
    // }
    while(I == J) { I = Uk(*rng); J = Uk(*rng); }
    double worstErr = Sigma[I][J].cor - Sigma[I][J].targetCor;


    double *P = x + I * N, *Q = x + J * N;
    INT u, v;
    bool negativeWorstErr = worstErr < 0;
    while(true)
    {
      // u = U(*rng); v = U(*rng);
      u = v = 0;
      while(u == v) {u = U(*rng); v = U(*rng);}
      bool haveSameSign = sameSign(P[u] - P[v], Q[u] - Q[v]);
      if(negativeWorstErr != haveSameSign) break;
    }


    INT col = I;
    // if(B(*rng)) col = J;
    double costDiff = 0;
    double &colu = X.at(u, col), &colv = X.at(v, col);
    wouldBeCor.resize(0);
    for(INT i = 0; i < K; ++i)
    {
      twoCor *a = nullptr;
      if(i < col) a = &Sigma[i][col];
      else if (i > col) a = &Sigma[col][i];
      else continue;
      double iu = X.at(u, i), iv = X.at(v, i);
      double tentativeCor = a->cor + (iu - iv) * (colv - colu);
      costDiff += std::pow(tentativeCor - a->targetCor, 2) -
        std::pow(a->cor - a->targetCor, 2);
      wouldBeCor.push_back(mem(tentativeCor, &a->cor));
    }


    if(costDiff < 0)
    {
      gap = 0;
      std::swap(colu, colv);
      for(INT i = 0, iend = wouldBeCor.size(); i < iend; ++i)
        *wouldBeCor[i].place = wouldBeCor[i].val;
    }
    else ++gap;
  }


  recover<double> (&X[0], N, K, shift, multiplier);
  for(INT i = 0, iend = K - 1; i < iend; ++i)
  {
    for(INT j = i + 1; j < K; ++j)
    {
      Xcor.at(j, i) = Sigma[i][j].cor;
      Xcor.at(i, j) = Sigma[i][j].cor;
    }
  }
}
*/




// [[Rcpp::export]]
List postSimOpt(NumericMatrix X,
                NumericMatrix cor,
                NumericMatrix Xcor = NumericMatrix(),
                NumericVector acceptProb = 1.0,
                IntegerVector seed = 123,
                int convergenceTail = 10000)
{
  INT N = X.nrow(), K = X.ncol();
  NumericMatrix X_(N, K);
  std::copy(X.begin(), X.end(), X_.begin());


  vec<double*> xptr(K);
  for(INT i = 0; i < K; ++i) xptr[i] = &X_[0] + i * N;
  double **x = &xptr[0];


  vec<double> shift, multiplier;
  normalize<double> (&x[0][0], N, K, shift, multiplier);
  if(Xcor.ncol() - K != 0 and Xcor.size() > 1)
  {
    Rcout << "`Xcor` has wrong number of columns. Quit.\n";
    return List::create();
  }
  if(cor.ncol() - K != 0)
  {
    Rcout << "`cor` has wrong number of columns. Quit.\n";
    return List::create();
  }


  RNG rengine, *rng = &rengine;
  if(seed.size() == 1) rng->seed(seed[0]);
  else
  {
    if(seed.size() * sizeof(int) != sizeof(RNG))
    {
      Rcerr << "`seed` should be either an integer or an integer vector of size 626. Quit.\n";
      return List::create();
    }
    rng = (RNG*)&seed[0];
  }


  // Find initial worst entries in correlation matrix.
  vec<twoCor> sigma(K * K);
  vec<twoCor*> sigmacol(K);
  for(INT i = 0; i < K; ++i) sigmacol[i] = &sigma[0] + K * i;
  twoCor **Sigma = &sigmacol[0];




  // If the current correlations are not given, calculate them.
  if(Xcor.size() > 1)
  {
    for(INT i = 0, iend = K - 1; i < iend; ++i)
    {
      double *XcorV = &Xcor[0] + i * K;
      double *targetCorV = &cor[0] + i * K;
      for(INT j = i + 1; j < K; ++j)
      {
        Sigma[i][j].cor = XcorV[j];
        Sigma[i][j].targetCor = targetCorV[j];
      }
    }
  }
  else
  {
    for(INT i = 0, iend = K - 1; i < iend; ++i)
    {
      double *targetCorV = &cor[0] + i * K;
      for(INT j = i + 1; j < K; ++j)
      {
        Sigma[i][j].cor = std::inner_product(
          &x[i][0], &x[i][0] + N, &x[j][0], 0.0);
        Sigma[i][j].targetCor = targetCorV[j];
      }
    }
  }




  INT gap = 0;
  std::uniform_int_distribution<INT> U(0, N - 1);
  std::bernoulli_distribution B(0.5);
  std::uniform_real_distribution<double> unif(0, 1);
  vec<mem> wouldBeCor(K - 1);


  vec<double> prob(acceptProb.begin(), acceptProb.end());
  std::reverse(prob.begin(), prob.end());
  std::partial_sum(prob.begin(), prob.end(), prob.begin());
  prob.back() = 1;
  vec<corIndDiff> nnCandidates(prob.size());
  int nn = nnCandidates.size();


  while(gap < convergenceTail)
  {
    double worstErr = 0;
    INT I = 0, J = 0;
    for(int i = 0; i < nn; ++i) nnCandidates[i].corDiff = 0;
    for(int i = 0, iend = K - 1; i < iend; ++i)
    {
      for(int j = i + 1, jend = K; j < jend; ++j)
      {
        push(i, j, Sigma[i][j].cor - Sigma[i][j].targetCor,
             &nnCandidates[0], nn);
      }
    }


    int chosen = std::lower_bound(
      prob.begin(), prob.end(), unif(*rng)) - prob.begin();
    I = nnCandidates[chosen].I;
    J = nnCandidates[chosen].J;
    worstErr = nnCandidates[chosen].corDiff;


    double *P = x[I], *Q = x[J];
    INT u, v;
    bool negativeWorstErr = worstErr < 0;
    while(true)
    {
      u = U(*rng); v = U(*rng);
      while(u == v) {u = U(*rng); v = U(*rng);}
      bool haveSameSign = sameSign(P[u] - P[v], Q[u] - Q[v]);
      if(negativeWorstErr != haveSameSign) break;
    }


    // u, v can reduce the error in the worst entry of correlation matrix,
    // but do they also reduce the cost function value?
    // double *col = I * N + x;
    INT col = I;
    if(B(*rng)) col = J; // swap() elements would apply to this column.
    double costDiff = 0;
    // double &colu = X_(u, col), &colv = X_(v, col);
    double &colu = x[col][u], &colv = x[col][v];
    wouldBeCor.resize(0);
    for(INT i = 0; i < K; ++i)
    {
      twoCor *a = nullptr;
      if(i < col) a = &Sigma[i][col];
      else if (i > col) a = &Sigma[col][i];
      else continue;
      double iu = x[i][u], iv = x[i][v];
      double tentativeCor = a->cor + (iu - iv) * (colv - colu);
      costDiff += std::pow(tentativeCor - a->targetCor, 2) -
        std::pow(a->cor - a->targetCor, 2);
      wouldBeCor.push_back(mem(tentativeCor, &a->cor));
    }


    if(costDiff < 0)
    {
      gap = 0;
      std::swap(colu, colv);
      for(INT i = 0, iend = wouldBeCor.size(); i < iend; ++i)
        *wouldBeCor[i].place = wouldBeCor[i].val;
    }
    else ++gap;
  }


  NumericMatrix corNew(K, K);
  for(INT i = 0; i < K; ++i)
  {
    corNew(i, i) = 1;
    for(INT j = i + 1; j < K; ++j)
    {
      corNew(j, i) = Sigma[i][j].cor;
      corNew(i, j) = Sigma[i][j].cor;
    }
  }
  return List::create(Named("X") = X_, Named("cor") = corNew);
}



















































