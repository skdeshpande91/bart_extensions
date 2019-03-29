//
//  uni_bartFit.cpp
//  
//
//  Created by Sameer Deshpande on 1/16/19.
//

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <algorithm>

#include "rng.h"
#include "tree.h"
#include "info.h"
#include "funs.h"
#include "bd.h"
#include "tree_prior.h"

#include <stdio.h>



using namespace Rcpp;
using namespace arma;


// [[Rcpp::export]]
Rcpp::List uni_bartFit(arma::vec Y,
                 arma::mat X,
                 arma::mat X_pred,
                 Rcpp::List xinfo_list,
                 int burn = 250, int nd = 1000, int m = 200, double kappa = 2, double nu = 3, double var_prob = 0.9)
{
  // Random number generator, used in all draws.
  Rcpp::Rcout << "Entered uni_bartFit" << endl;

  RNGScope scope;
  RNG gen;
  
  size_t n_obs = X.n_rows;
  size_t n_pred = X_pred.n_rows;
  size_t p = X.n_cols;
  size_t q = 1; // for univariate BART we have only one outcome
  
  size_t n = Y.size(); // Should always be n_obs * q
  /*
  Rcpp::Rcout << "  n_obs = " << n_obs << endl;
  Rcpp::Rcout << "  n_pred = " << n_pred << endl;
  Rcpp::Rcout << "  p = " << p << endl;
  Rcpp::Rcout << "  q = " << q << endl;
  */

  // always center and scale the y's
  arma::vec Y_orig = Y;
  double y_col_mean = arma::mean(Y);
  double y_col_sd = arma::stddev(Y);
  Y -= y_col_mean;
  Y /= y_col_sd;
  //Rcpp::Rcout << "Centered and scaled Y" << endl;
  

  // create pointer for x, y, and x_pred
  double* y_ptr = new double[n];
  double* x_ptr = new double[n_obs*p];
  double* x_pred_ptr = new double[n_pred*p];
  
  for(size_t i = 0; i < n_obs; i++){
    y_ptr[i] = Y(i);
    for(size_t j = 0; j < p; j++){
      x_ptr[j + i*p] = X(i,j);
    }
  }
  
  for(size_t i = 0; i < n_pred; i++){
    for(size_t j = 0; j < p; j++){
      x_pred_ptr[j + i*p] = X_pred(i,j);
    }
  }
  Rcpp::Rcout << "  created y_ptr, x_ptr, x_pred_ptr" << endl;

  // Read-in and format the cut-points
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
  Rcpp::Rcout << "Created xi" << endl;
  
  double omega = 1.0; // the precision
  double S = 0.0 ; // sum of squares of residuals
  
  // Set up the prior hyper-paramters
  pinfo pi;
  pi.pbd = 1.0;
  pi.pb = 0.5;
  pi.alpha = 0.95;
  pi.beta = 2.0;
  pi.nu = nu;
  
  //  sigma_mu, sigma_hat, and lambda will all be of length 1.
  pi.sigma_mu.clear();
  pi.sigma_mu.reserve(q);
  
  pi.sigma_hat.clear();
  pi.sigma_hat.reserve(q);
  
  pi.lambda.clear();
  pi.lambda.reserve(q);
  
  double chisq_quantile = 0.0;
  Function qchisq("qchisq");
  NumericVector tmp_quantile = qchisq(Named("p") = 1.0 - var_prob, Named("df") = nu);
  chisq_quantile = tmp_quantile[0];
  
  pi.sigma_hat[0] = arma::stddev(Y); // should be 1
  pi.lambda[0] = (pi.sigma_hat[0] * pi.sigma_hat[0] * chisq_quantile)/nu;
  pi.sigma_mu[0] = (Y.max() - Y.min())/(2.0 * kappa * sqrt( (double) m));
  
  Rcpp::Rcout << "pi.sigma_hat[0] = " << pi.sigma_hat[0] << endl;
  Rcpp::Rcout << "pi.lambda[0] = " << pi.lambda[0] << endl;
  Rcpp::Rcout << "pi.sigma_mu[0] = " << pi.sigma_mu[0] << endl;
  
  // trees, data, and residuals
  std::vector<tree> t_vec(m);
  
  double* allfit = new double[n]; // allfit[i] is fit of m trees for observation i
  double* allfit_pred = new double[n_pred];
  double* r_full = new double[n]; // r_full[i] is full residual for observation i
  double* r_partial = new double[n_obs]; // r_partial[i] is partial residual for observation i
  double* ftemp = new double[n_obs];
  double* ftemp_pred = new double[n_pred];
  double ybar = arma::mean(Y); // used to set initial values of allfit. should be 0
  for(size_t t = 0; t < m; t++){
    t_vec[t].setm( ybar/ ( (double) m));
  }
  for(size_t i = 0; i < n_obs; i++){
    allfit[i] = ybar;
    r_full[i] = y_ptr[i] - allfit[i];
  }
  // initialize values in r_partial and ftemp. These initial values don't matter much
  for(size_t i = 0; i < n_obs; i++){
    r_partial[i] = 0.0;
    ftemp[i] = 0.0;
  }
  
  dinfo di;
  di.n = n_obs;
  di.p = p;
  di.q = q;
  di.k = 0; // this will never be changed in this function
  di.x = &x_ptr[0];
  di.r_f = &r_full[0];
  di.r_p = &r_partial[0];
  
  dinfo dip;
  dip.n = n_pred;
  dip.p = p;
  dip.q = q;
  dip.k = 0;
  dip.x = &x_pred_ptr[0];
  
  Rcpp::Rcout << "  initialized pointers for residuals" << endl;
  
  // We run the main procedure using the centered and scaled responses.
  // we must output to the original scale of the data

  arma::mat train_samples = arma::zeros<arma::mat>(n_obs, nd); // train_samples(i,iter) = y_col_mean + y_col_sd * allfit[i];
  arma::mat test_samples = arma::zeros<arma::mat>(n_pred, nd); // test_samples(i,iter) = y_col_mean + y_col_sd * allfit_pred[i];
  arma::vec omega_samples = zeros<vec>(nd); // omega_samples(iter) = std_omega_samples(iter)/y_col_sd
  arma::vec sigma_samples = zeros<vec>(nd); // sigma_samples(iter) = 1.0/sqrt(omega_samples(iter));
  
  // some diagnostic stuff
  arma::vec S_samples = zeros<vec>(nd+burn); // residual sum of squares
  arma::mat alpha_samples = zeros<mat>(nd+burn, m); // alpha_samples(iter, t) is alpha for iteration iter, tree t
  arma::mat tree_size_samples = zeros<mat>(nd+burn, m); // tree_size(iter,t) is size of tree t in iteration iter
  
  Rcpp::Rcout << "Starting MCMC" << endl;
  time_t tp;
  int time1 = time(&tp);
  for(size_t iter = 0; iter < (nd + burn); iter++){
    if(iter < burn & iter%50 == 0) Rcpp::Rcout << "  MCMC Iteration: " << iter << " of " << nd + burn << "; Burn-in" << endl;
    else if( (iter > burn & iter%50 == 0) || (iter == burn)) Rcpp::Rcout << "  MCMC Iteration: " << iter << " of " << nd + burn << "; Sampling" << endl;
    
    if(iter%100 == 0) Rcpp::checkUserInterrupt(); // check for user interruption every 100 iterations
    
    // Updating trees
    for(size_t t = 0; t < m; t++){
      fit(t_vec[t], xi, di, ftemp); // Get the current fit of the tree
      for(size_t i = 0; i < n_obs; i++){
        if(ftemp[i] != ftemp[i]){
          Rcpp::Rcout << "tree " << t << " observation " << i << endl;
          Rcpp::stop("nan in ftemp");
        } // closes if checking whether ftemp[i] is nan
        allfit[i] = allfit[i] - ftemp[i];
        r_partial[i] = y_ptr[i] - allfit[i];
      } // closes loop over observations updating allfit and r_partial
      
      
      alpha_samples(iter,t) = bd_uni(t_vec[t], omega, xi, di, pi, gen); // Do the birth/death move
      drmu_uni(t_vec[t], omega, xi, di, pi, gen); // Draw the new values of the mu parameters
      fit(t_vec[t], xi, di, ftemp); // Update the fit
      
      for(size_t i = 0; i < n_obs; i++){
        if(ftemp[i] != ftemp[i]){
          Rcpp::Rcout << "tree " << t << " observation " << i << endl;
          Rcpp::stop("nan in ftemp");
        } // closes if checking whether ftemp[i] is nan
        allfit[i] += ftemp[i];
        r_full[i] = y_ptr[i] - allfit[i];
      } // closes loop over observations updating allfit and r_full
      tree_size_samples(iter, t) = t_vec[t].treesize();
    } // closes loop over trees
    
    // Now we update omega
    // A priori sigma2 ~ (nu * lambda)/chisq_nu = IG(nu/2, (nu * lambda)/2)
    // A posteriori sigma2 ~ IG( (nu + n_obs)/2, (nu*lambda + S)/2) = (nu * lambda + S) * inv_chisq_(nu + n_obs)
    S = 0.0;
    for(size_t i = 0; i < n_obs; i++) S += r_full[i] * r_full[i];
    omega = gen.chi_square(pi.nu + di.n)/(S + pi.nu * pi.lambda[0]);
    // now save the samples
    if(iter >= burn){
      for(size_t i = 0; i < n_obs; i++) train_samples(i,iter-burn) = y_col_sd * allfit[i] + y_col_mean;
      omega_samples(iter-burn) = omega/(y_col_sd * y_col_sd);
      sigma_samples(iter-burn) = y_col_sd * 1.0/sqrt(omega);
      
      // now get the out of sample predictions
      if(dip.n > 0){
        for(size_t i = 0; i < dip.n; i++) allfit_pred[i] = 0.0;// reset everything in allfit_pred
        // loop over trees, get the fit, and update allfit_pred
        for(size_t t = 0; t < m; t++){
          fit(t_vec[t], xi, dip, ftemp_pred); // got the fit of tree t
          for(size_t i = 0; i < n_pred; i++) allfit_pred[i] += ftemp_pred[i]; // update allfit_pred!
        }
        for(size_t i = 0; i < dip.n; i++) test_samples(i,iter-burn) = allfit_pred[i] * y_col_sd + y_col_mean; // now write to the final output
      }
    } // closes if checking that iter >= burn and we need to save the samples
  } // closes MCMC loop
  Rcpp::Rcout << "Finished MCMC" << endl;
  int time2 = time(&tp);
  Rcpp::Rcout << "time for MCMC: " << time2 - time1 << endl;
  
  // clean up some memory
  delete[] r_full;
  delete[] r_partial;
  delete[] allfit;
  delete[] allfit_pred;
  delete[] ftemp;
  delete[] ftemp_pred;
  
  delete[] x_ptr;
  delete[] x_pred_ptr;
  delete[] y_ptr;
  Rcpp::List results;
  results["Y"] = Y;
  results["X"] = X;
  results["Y_orig"] = Y;
  results["y_col_mean"] = y_col_mean;
  results["y_col_sd"] = y_col_sd;
  results["train_samples"] = train_samples;
  results["test_samples"] = test_samples;
  //results["omega_samples"] = omega_samples;
  results["sigma_samples"] = sigma_samples;
  results["alpha_samples"] = alpha_samples;
  results["tree_size_samples"] = tree_size_samples;
  results["s_samples"] = S_samples;

  return(results);
  
}
