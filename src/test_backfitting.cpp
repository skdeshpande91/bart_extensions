//
//  test_backfitting.cpp
//  This is meant to test the back-fitting procedure, specifically the functions
//  mu_posterior_slfm and bd_slfm
//
//  Created by Sameer Deshpande on 2/25/19.
//

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <vector>
#include <ctime>
#include <algorithm>
#include <stdio.h>
#include "rng.h"
#include "tree.h"
#include "info.h"
#include "funs.h"
#include "bd.h"
#include "tree_prior.h" // so that we can generate some random trees

using namespace Rcpp;
using std::endl;

// [[Rcpp::export]]
Rcpp::List test_backfitting(arma::mat Y,
                            arma::mat X,
                            arma::mat X_pred,
                            Rcpp::List xinfo_list,
                            arma::mat Phi_init, arma::vec sigma_init, // only for testing
                            int burn = 250, int nd = 1000,
                            int D = 5, int m = 200, double kappa = 3, double nu = 1, double var_prob = 0.9)
{
  Rcpp::RNGScope scope;
  RNG gen;
  // Get parameters about our data
  size_t n_obs = X.n_rows;
  size_t n_pred = X_pred.n_rows;
  size_t q = Y.n_cols;
  size_t p = X.n_cols;
  size_t n = Y.size(); // should be n_obs * q
  
  //Y.print();
  arma::mat Y_orig = Y; // just a copy of the Y that is initially passed into the function
                        // Center and scale Y
  arma::vec y_col_mean = arma::zeros<arma::vec>(q); // holds the original mean of each column of Y
  arma::vec y_col_sd = arma::zeros<arma::vec>(q); // holds the original sd of each column of Y
  arma::vec y_col_max = arma::zeros<arma::vec>(q); // holds the max of the scaled columns of Y
  arma::vec y_col_min = arma::zeros<arma::vec>(q); // holds the min of the scaled columns of Y
  
  prepare_y(Y, y_col_mean, y_col_sd, y_col_max, y_col_min);
  
  Rcpp::Rcout << "  Centered and scaled Y" << endl;
  
  // create arrays (i.e. pointers for X, Y, X_pred, and delta)
  double* y_ptr = new double[n];
  double* delta_ptr = new double[n];
  double* x_ptr = new double[n_obs * p];
  double* x_pred_ptr = new double[n_pred * p];
  
  for(size_t i = 0; i < n_obs; i++){
    for(size_t k = 0; k < q; k++){
      y_ptr[k + i*q] = Y(i,k);
      if(Y(i,k) == Y(i,k)) delta_ptr[k + i*q] = 1;
      else delta_ptr[k + i*q] = 0;
    }
    for(size_t j = 0; j < p; j++){
      x_ptr[j + i*p] = X(i,j);
    }
  }
  for(size_t i = 0; i < n_pred; i++){
    for(size_t j = 0; j < p; j++){
      x_pred_ptr[j + i*p] = X_pred(i,j);
    }
  }
  Rcpp::Rcout << "  Created pointers" << endl;
  
  // Read and format the cut-puts
  xinfo xi;
  xi.resize(p);
  for(size_t j = 0; j < p; j++){
    Rcpp::NumericVector tmp = xinfo_list[j];
    std::vector<double> tmp2;
    for(int jj = 0; jj < tmp.size(); jj++){
      tmp2.push_back(tmp[jj]);
    }
    xi[j] = tmp2;
  }
  Rcpp::Rcout << "  Created cut-points" << endl;
  
  // Introduce Phi and sigma
  //arma::mat Phi = arma::ones<arma::mat>(q,D); // Phi(k,d) tells us how much u_d contribute to f_k
  arma::mat Phi = Phi_init;
  //arma::vec sigma = arma::ones<arma::vec>(q); // sigma(k) is the residual SD for task k
  arma::vec sigma = sigma_init;
  
  
  // set up prior hyper-parameters
  pinfo pi;
  pi.pbd = 1.0; // probability of a birth/death move
  pi.pb = 0.5; // probability of a birth move given birth/death move occurs
  pi.alpha = 0.95;
  pi.beta = 2.0;
  pi.sigma_mu.clear(); // prior sd of the mu parameters for each tree in the D basis functions
  pi.sigma_mu.reserve(D);
  
  pi.sigma_hat.clear(); // over-estimate of the residual variance for each task
  pi.sigma_hat.reserve(q);
  
  pi.lambda.clear(); // scaling value for the scaled-inverse chi-square prior on residual variances
  pi.lambda.reserve(q);
  pi.nu = nu; // df of the scaled-inverse chi-square prior on residual variances
  
  double chisq_quantile = 0.0;
  Function qchisq("qchisq");
  NumericVector tmp_quantile = qchisq(Named("p") = 1.0 - var_prob, Named("df") = nu);
  chisq_quantile = tmp_quantile[0];
  for(size_t d = 0; d < D; d++){
    pi.sigma_mu[d] = (y_col_max.max() - y_col_min.min())/(2.0 * kappa * sqrt( (double) m));
  }
  for(size_t k = 0; k < q; k++){
    pi.sigma_hat[k] = 1.0; // every column has variance 1
    pi.lambda[k] = chisq_quantile/nu;
  }
  Rcpp::Rcout << "Finished setting pi" << endl;
  
  
  Rcpp::Rcout << "  pi.sigma_hat:" ;
  for(size_t k = 0; k < q; k++) Rcpp::Rcout << " " << pi.sigma_hat[k] ;
  Rcpp::Rcout << endl;
  
  Rcpp::Rcout << "  pi.lambda:" ;
  for(size_t k = 0; k < q; k++) Rcpp::Rcout << " " << pi.lambda[k] ;
  Rcpp::Rcout << endl;
  
  Rcpp::Rcout << "  pi.sigma_mu:" ;
  for(size_t d = 0; d < D; d++){
    Rcpp::Rcout << " " << pi.sigma_mu[d] ;
  }
  Rcpp::Rcout << endl;
  // [SKD]: If done correctly, pi.sigma_hat is 1 for every outcome, pi.lambda is constant, and pi.sigma_mu constant
  
  // Set up trees, data, and residuals
  std::vector<std::vector<tree> > t_vec(D, std::vector<tree>(m));
  
  double* allfit = new double[n]; // allfit[k + i*q] is estimated fit of i^th observation, k^th task
  double* ufit = new double[n_obs*D]; // ufit[d + i*D] is estimated fit of i^th observation, d^th basis
  double* ftemp = new double[n_obs]; // temporariliy holds current fit of a single tree
  
  // note that all of the columns have been centered
  // will set the terminal node parameter of each tree to be 0 to begin
  for(size_t d = 0; d < D; d++){
    for(size_t t = 0; t < m; t++){
      t_vec[d][t].setm(0.0);
    }
  }
  for(size_t i = 0; i < n_obs; i++){
    for(size_t d = 0; d < D; d++){
      ufit[d + i*D] = 0.0;
    }
    for(size_t k = 0; k < q; k++){
      allfit[k + i*q] = 0.0; // remember that all of trees evaluate to 0 initially
    }
    ftemp[i] = 0.0;
  }
  
  dinfo_slfm di;
  di.p = p;
  di.n = n_obs;
  di.q = q;
  di.d = 0;
  di.x = &x_ptr[0];
  di.y = &y_ptr[0];
  di.delta = &delta_ptr[0];
  di.af = &allfit[0];
  di.uf = &allfit[0];
  
  dinfo_slfm dip;
  dip.p = p;
  dip.n = n_obs;
  dip.q = q;
  dip.d = 0;
  dip.x = &x_pred_ptr[0];
  
  Rcpp::Rcout << "  Created arrays to hold residuals, etc." << endl;
  
  double mu_bar = 0.0;
  double V = 0.0;
  // draw a tree now
  size_t t = 0; // only work with the first tree
  for(size_t d = 0; d < 1; d++){
    di.d = d;
    // draw a tree
    draw_tree(t_vec[d][t], xi, pi.alpha, pi.beta, gen);
    Rcpp::Rcout << "Tree for d = " << d << endl;
    //t_vec[d][t].pr(true); // this prints out the tree
    
    // need to call allsuff
    tree::npv bnv;
    std::vector<sinfo> sv;
    allsuff(t_vec[d][t], xi, di, bnv, sv);
    Rcpp::Rcout << "Got sufficient statistics for each terminal node" << endl;
    for(size_t l = 0; l < bnv.size(); l++){
      Rcpp::Rcout << "  terminal node " << l << " contains :" << endl;
      for(size_t ii = 0; ii < sv[l].n; ii++){
        Rcpp::Rcout << " " << sv[l].I[ii];
      }
      Rcpp::Rcout << endl;
      
      mu_posterior_slfm(mu_bar, V, Phi, sigma, sv[l], di, pi.sigma_mu[d]);
      Rcpp::Rcout << "M = " << mu_bar << "  V = " << V << endl;
      
    }
  }
  
  
  // Need to return something
  
  Rcpp::List results;
  results["Y_orig"] = Y_orig;
  results["Y"] = Y;
  results["y_col_max"] = y_col_max;
  results["y_col_min"] = y_col_min;
  results["y_col_mean"] = y_col_mean;
  results["y_col_sd"] = y_col_sd;
  return(results);
}
