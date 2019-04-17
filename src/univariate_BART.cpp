//
//  univariate_BART
//    Uses new prior info classes.
//    Allows for weighted likelihood.
//  Created by Sameer Deshpande on 16 April 2019
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
Rcpp::List univariate_BART(arma::vec Y,
                           arma::mat X,
                           arma::mat X_pred,
                           Rcpp::List xinfo_list,
                           double weight = 1.0,
                           int burn = 250, int nd = 1000,
                           int m = 200, double kappa = 2,
                           double nu = 3, double var_prob = 0.9,
                           bool verbose = false)
{
  // Random number generator, used in all draws.
  if(verbose == true) Rcpp::Rcout << "Entered univariate_BART" << endl;

  RNGScope scope;
  RNG gen;
  
  size_t n_obs = X.n_rows;
  size_t n_pred = X_pred.n_rows;
  size_t p = X.n_cols;
  size_t q = 1; // for univariate BART we have only one outcome
  
  size_t n = Y.size(); // Should always be n_obs * q


  // always center and scale the y's
  arma::vec Y_orig = Y;
  double y_col_mean = arma::mean(Y);
  double y_col_sd = arma::stddev(Y);
  Y -= y_col_mean;
  Y /= y_col_sd;
  //Rcpp::Rcout << "Centered and scaled Y" << endl;
  

  // create pointer for x, y, and x_pred
  double* y_ptr = new double[n];
  double* delta_ptr = new double[n];
  double* x_ptr = new double[n_obs*p];
  double* x_pred_ptr = new double[n_pred*p];
  
  for(size_t i = 0; i < n_obs; i++){
    y_ptr[i] = Y(i);
    for(size_t j = 0; j < p; j++) x_ptr[j + i*p] = X(i,j);
    if(Y(i) == Y(i)) delta_ptr[i] = 1; // true unless Y(i) is nan, which is how missing data is parsed
    else delta_ptr[i] = 0; // indicates we are missing observation for Y(i)
  }
  
  for(size_t i = 0; i < n_pred; i++){
    for(size_t j = 0; j < p; j++) x_pred_ptr[j + i*p] = X_pred(i,j);
  }
  if(verbose == true) Rcpp::Rcout << "  created y_ptr, x_ptr, x_pred_ptr" << endl;

  // Read-in and format the cut-points
  xinfo xi;
  xi.resize(p);
  for(size_t j = 0; j < p; j++){
    Rcpp::NumericVector tmp = xinfo_list[j];
    std::vector<double> tmp2;
    for(int jj = 0; jj < tmp.size(); jj++) tmp2.push_back(tmp[jj]);
    xi[j] = tmp2;
  }
  if(verbose == true) Rcpp::Rcout << "Created xi" << endl;
  
  double sigma = 1.0;// will track the residual variances
  
  
  // trees and pointers for fit and residuals
  std::vector<tree> t_vec(m);
  
  double* allfit = new double[n]; // allfit[k + i*q] will be i^th prediction for k^th outcome
  double* allfit_pred = new double[n_pred*q];
  double* r_full = new double[n]; // r_full[k+i*q] will be i^th residual for k^th outcome
  double* r_partial = new double[n_obs]; // temporarily holds partial residuals
  
  double* ftemp = new double[n_obs]; // temporarily holds fit of a single tree
  double* ftemp_pred = new double[n_pred];

  // assuming that all Y's have been centered by this point, can assing mu = 0.0
  for(size_t t = 0; t < m; t++) t_vec[t].setm(0.0); //
  
  for(size_t i = 0; i < n_obs; i++){
    allfit[i] = 0.0;
    r_full[i] = y_ptr[i] - allfit[i];
  }
  
  // initialize values in r_partial and ftemp. These initial values don't matter much at all
  for(size_t i = 0; i < n_obs; i++){
    r_partial[i] = 0.0;
    ftemp[i] = 0.0;
  }
  // initialize allfit_pred and ftemp_pred
  for(size_t i = 0; i < n_pred; i++){
    ftemp_pred[i] = 0.0;
    allfit_pred[i] = 0.0;
  }
  
  // We can now create the data_info, tree_prior_info, and sigma_prior_info objects
  
  // Stuff for trees
  tree_prior_info tree_pi;
  tree_pi.pbd = 1.0;
  tree_pi.pb = 0.5;
  tree_pi.alpha = 0.95;
  tree_pi.beta = 2.0;
  tree_pi.sigma_mu = (Y.max() - Y.min())/(2.0 * kappa * sqrt( (double) m)); // this is the wbart default
  tree_pi.r_p = &r_partial[0]; // whenever we update trees, we need the partial residuals
  
  sigma_prior_info sigma_pi;
  
  sigma_pi.sigma_hat = 1.0; // initial over-estimate of the residual deviation
  double chisq_quantile = 0.0;
  Function qchisq("qchisq");
  NumericVector tmp_quantile = qchisq(Named("p") = 1.0 - var_prob, Named("df") = nu);
  chisq_quantile = tmp_quantile[0];
  sigma_pi.lambda = (sigma_pi.sigma_hat * sigma_pi.sigma_hat * chisq_quantile)/nu;
  sigma_pi.nu = nu;

  if(verbose == true){
    Rcpp::Rcout << "sigma_pi.sigma_hat = " << sigma_pi.sigma_hat << endl;
    Rcpp::Rcout << "sigma_pi.lambda = " << sigma_pi.lambda << endl;
    Rcpp::Rcout << "tree_pi.sigma_mu = " << tree_pi.sigma_mu << endl;
  }
 
  data_info di;
  di.n = n_obs;
  di.p = p;
  di.q = q;
  di.x = &x_ptr[0];
  di.y = &y_ptr[0];
  di.r_f = &r_full[0]; // track full residuals in data_info. This comes in handy when we update sigma
  di.delta = &delta_ptr[0];
  di.weight = weight;
  
  data_info dip;
  dip.n = n_pred;
  dip.p = p;
  dip.q = q;
  dip.x = &x_pred_ptr[0];
  
  if(verbose == true) Rcpp::Rcout << "  Created arrays to hold residuals etc." << endl;
  
  // created containers for output
  arma::mat f_train_samples = arma::zeros<arma::mat>(n_obs, nd); // f_train_samples(i,iter) = y_col_mean + y_col_sd * allfit[i];
  arma::mat f_test_samples = arma::zeros<arma::mat>(n_pred, nd); // f_test_samples(i,iter) = y_col_mean + y_col_sd * allfit_pred[i];
  arma::vec sigma_samples = arma::zeros<arma::vec>(nd); // sigma_samples(iter) = sigma * y_col_sd;

  // these are strictly for diagnostics
  arma::mat alpha_samples = arma::zeros<arma::mat>(m, nd+burn); // alpha_samples(t, iter) is the MH acceptable probability for tree t
  arma::mat tree_size_samples = arma::zeros<arma::mat>(m, nd+burn); //

  Rcpp::Rcout << "Starting MCMC" << endl;
  time_t tp;
  int time1 = time(&tp);
  for(size_t iter = 0; iter < (nd + burn); iter++){
    if(verbose == true){
      if(iter < burn & iter%50 == 0) Rcpp::Rcout << "  MCMC Iteration: " << iter << " of " << nd + burn << "; Burn-in" << endl;
      else if( (iter > burn & iter%50 == 0) || (iter == burn)) Rcpp::Rcout << "  MCMC Iteration: " << iter << " of " << nd + burn << "; Sampling" << endl;
    }
    if(iter%100 == 0) Rcpp::checkUserInterrupt(); // check for user interruption every 100 iterations
    
    // Updating trees
    for(size_t t = 0; t < m; t++){
      fit(t_vec[t], xi, di, ftemp); // Get the current fit of the tree
      for(size_t i = 0; i < n_obs; i++){
        if(ftemp[i] != ftemp[i]){
          Rcpp::Rcout << "tree " << t << " observation " << i << endl;
          Rcpp::stop("nan in ftemp");
        } // closes if checking whether ftemp[i] is nan
        allfit[i] = allfit[i] - ftemp[i]; // temporarily remove fit of tree t from allfit
        r_partial[i] = y_ptr[i] - allfit[i]; // allfit contains fit of (m-1) trees so this is the correct value of r_partial
      } // closes loop over observations updating allfit and r_partial
      
      alpha_samples(t, iter) = bd_uni(t_vec[t], sigma, xi, di, tree_pi,gen); // do the birth/death move
      drmu_uni(t_vec[t], sigma, xi, di, tree_pi, gen); // Draw the new mu parameters
      fit(t_vec[t], xi, di, ftemp); // Update the fit
      
      for(size_t i = 0; i < n_obs; i++){
        if(ftemp[i] != ftemp[i]) Rcpp::stop("nan in ftemp");
        allfit[i] += ftemp[i]; // add fit of tree t back to allfit
        r_full[i] = y_ptr[i] - allfit[i]; // update the full residual
      }
      tree_size_samples(t, iter) = t_vec[t].treesize(); //
    } // closes loop over trees
    
    // Now we can update sigma
    update_sigma_uni(sigma, sigma_pi, di, gen);
    
    // save samples
    if(iter >= burn){
      // save training fit
      for(size_t i = 0; i < n_obs; i++) f_train_samples(i,iter-burn) = y_col_mean + y_col_sd * allfit[i];
      
      // save sigm
      sigma_samples(iter-burn) = y_col_sd * sigma;
      
      // save test output. start by clearing all of the elements in allfit_pred and ftemp_pred
      for(size_t i = 0; i < n_pred; i++){
        ftemp_pred[i] = 0;
        allfit_pred[i] = 0;
      }
      for(size_t t = 0; t < m; t++){
        fit(t_vec[t], xi, dip, ftemp_pred); // gets fit of tree t for all of the test inputs
        for(size_t i = 0; i < n_pred; i++) allfit_pred[i] += ftemp_pred[i]; // update allfit_pred;
      }
      // at this point allfit_pred contains total fit from the m trees on the test inputs
      for(size_t i = 0; i < n_pred; i++) f_test_samples(i,iter-burn) = y_col_mean + y_col_sd * allfit_pred[i];
      
    } // closes if checking that iter > burn and that we should save samples
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
  delete[] delta_ptr;
  
  Rcpp::List results;
  
  results["f_train_samples"] = f_train_samples;
  results["f_test_samples"] = f_test_samples;
  results["sigma_samples"] = sigma_samples;
  //results["alpha_samples"] = alpha_samples
  //results["tree_depth_samples"] = tree_depth_samples;
  results["time"] = time2 - time1;


  return(results);
  
}
