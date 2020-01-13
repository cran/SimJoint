// [[Rcpp::depends(RcppArmadillo)]]
# include <RcppArmadillo.h>


// [[Rcpp::export]]
void decor(arma::mat &seedMat)
{
  int ncol = seedMat.n_cols, nrow = seedMat.n_rows;
  double *M = &seedMat[0];
  arma::colvec S(ncol);
  for(int i = 0; i < ncol; ++i)
    S[i] = std::accumulate(M + i * nrow, M + i * nrow + nrow, 0.0);


  arma::mat X(ncol, ncol);
  arma::colvec Y(ncol);
  arma::colvec sol(ncol);
  for(int i = 1; i < ncol; ++i)
  {
    X.set_size(i, i);
    Y.set_size(i);
    sol.set_size(i);
    for(int j = 0; j < i; ++j)
    {
      for(int k = 0; k < i; ++k) X.at(j, k) = nrow * seedMat.at(k, j) - S[j];
      Y[j] = (S[i] - std::accumulate(M + i * nrow, M + i * nrow + i, 0.0)) *
        S[j] - nrow * std::inner_product(
            M + i * nrow + i, M + i * nrow + nrow, M + j * nrow + i, 0.0);
    }


    arma::solve(sol, X, Y);
    S[i] += std::accumulate(&sol[0], &sol[0] + i, 0.0) -
      std::accumulate(M + i * nrow, M + i * nrow + i, 0.0);
    std::copy(sol.begin(), sol.end(), M + i * nrow);
  }
}






