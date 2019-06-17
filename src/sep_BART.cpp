//
//  sep_BART.cpp
//    Uses new prior info classes
//    Allows for weighted likelihood
//  Created by Sameer Deshpande on 17 April 2019
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
Rcpp::List sep_BART(arma::mat Y,
                    arma::mat X,
                    arma::mat X_pred,
                    Rcpp::List xinfo_list,
                    double weight = 1.0,
                    int burn = 250, int nd = 1000,
                    int m = 200, double kappa = 2,
                    double nu = 3, double var_prob = 0.9,
                    bool verbose = false)
{
  if(verbose == true) Rcpp::Rcout << "Entering sep_BART" << endl;
  RNGScope scope;
  RNG gen;
  
  size_t n_obs = X.n_rows;
  size_t n_pred = X_pred.n_rows;
  size_t p = X.n_cols;
  size_t q = Y.n_cols;
  size_t n = Y.size();
  
  // Center and scale Y. Also compute columns mean, sd, min, and max
  std::vector<double> y_col_mean(q); // holds the original mean of each column of Y
  std::vector<double> y_col_sd(q); // holds the original sd of each column of Y
  std::vector<double> y_col_max(q); // holds the max of the scaled columns of Y
  std::vector<double> y_col_min(q); // holds the min of the scaled columns of Y
  
  prepare_y(Y, y_col_mean, y_col_sd, y_col_max, y_col_min);
  
  if(verbose == true) Rcpp::Rcout << "  Centered and scaled Y" << std::endl;
  
  // create pointers for x, y, x_pred, and delta
  double* y_ptr = new double[n];
  double* delta_ptr = new double[n];
  double* x_ptr = new double[n_obs*p];
  double* x_pred_ptr = new double[n_pred * p];
  
  for(size_t i = 0; i < n_obs; i++){
    for(size_t k = 0; k < q; k++){
      y_ptr[k + i*q] = Y(i,k);
      if(Y(i,k) == Y(i,k)) delta_ptr[k + i*q] = 1;
      else delta_ptr[k + i*q] = 0;
    }
    for(size_t j = 0; j < p; j++) x_ptr[j + i*p] = X(i,j);
  }
  for(size_t i = 0; i < n_pred; i++){
    for(size_t j = 0; j < p; j++) x_pred_ptr[j + i*p] = X_pred(i,j);
  }
  if(verbose == true) Rcpp::Rcout << "  Created pointers for X, X_pred, and Y" << std::endl;
  
  // Read and format the cutpoints
  xinfo xi;
  xi.resize(p);
  for(size_t j = 0; j < p; j++){
    Rcpp::NumericVector tmp = xinfo_list[j];
    std::vector<double> tmp2;
    for(int jj = 0; jj < tmp.size(); jj++) tmp2.push_back(tmp[jj]);
    xi[j] = tmp2;
  }
  if(verbose == true) Rcpp::Rcout << "  Created cutpoints" << std::endl;
  
  // Initialize sigma
  std::vector<double> sigma(q); // vector of residual standard deviations
  for(size_t k = 0; k < q; k++) sigma[k] = 1.0;
  Rcpp::Rcout << "Initial values of sigma: " ;
  for(size_t k = 0; k < q; k++) Rcpp::Rcout << " " << sigma[k] ;
  Rcpp::Rcout << endl;
  
  
  // Initialize trees and pointers for fit and residuals
  std::vector<std::vector<tree> > t_vec(q, std::vector<tree>(m));
  
  double* allfit = new double[n]; // allfit[k + i*q] is estimated fit of k^th task for i^th observation
  double* allfit_pred = new double[n_pred*q];
  double* ftemp = new double[n_obs]; // temporariliy holds fit a single tree
  double* ftemp_pred = new double[n_pred];
  
  double* r_full = new double[n]; // holds all residuals for training observations
  double* r_partial = new double[n_obs]; // holds partial residuals
  //double* r_partial = new double[n_obs * q]; // holds partial residuals for all observations
  
  for(size_t k = 0; k < q; k++){
    for(size_t t = 0; t < m; t++) t_vec[k][t].setm(0.0);
    for(size_t i = 0; i < n_obs; i++){
      allfit[k + i*q] = 0.0;
      if(delta_ptr[k + i*q] == 1) r_full[k + i *q] = y_ptr[k + i * q]  - allfit[k + i*q];
    }
    for(size_t i = 0; i < n_pred; i++) allfit_pred[k + i*q] = 0.0;
  }
  for(size_t i = 0; i < n_obs; i++){
    ftemp[i] = 0.0;
    r_partial[i] = 0.0;
  }
  for(size_t i = 0; i < n_pred; i++) ftemp_pred[i] = 0.0;

  
  // The f_k's have an independent but not identical BART priors, create vectors of tree_prior_info, sigma_prior
  std::vector<tree_prior_info> tree_pi(q);
  std::vector<sigma_prior_info> sigma_pi(q);

  double chisq_quantile = 0.0;
  Function qchisq("qchisq"); // use R's built-in quantile function
  Rcpp::NumericVector tmp_quantile = qchisq(Named("p") = 1.0 - var_prob, Named("df") = nu);
  chisq_quantile = tmp_quantile[0];
  for(size_t k = 0; k < q; k++){
    tree_pi[k].pbd = 1.0;
    tree_pi[k].pb = 0.5;
    tree_pi[k].alpha = 0.95;
    tree_pi[k].beta = 2.0;
    tree_pi[k].sigma_mu = (y_col_max[k] - y_col_min[k])/(2.0 * kappa * sqrt( (double) m));
    tree_pi[k].r_p = &r_partial[0];
    // possibly problematic that we have every element of tree_pi tracking r_partial but we'll leave it for now
    
    sigma_pi[k].sigma_hat = 1.0; // initial over-estimate of residual standard deviation
    sigma_pi[k].nu = nu;
    sigma_pi[k].lambda = (sigma_pi[k].sigma_hat * sigma_pi[k].sigma_hat * chisq_quantile)/nu;
  }
  
  if(verbose == true){
    Rcpp::Rcout << "  sigma_hat: " ;
    for(size_t k = 0; k < q; k++) Rcpp::Rcout << " " << sigma_pi[k].sigma_hat;
    Rcpp::Rcout << std::endl;
    Rcpp::Rcout << "  lambda: ";
    for(size_t k = 0; k < q; k++) Rcpp::Rcout << " " << sigma_pi[k].lambda;
    Rcpp::Rcout << std::endl;
    Rcpp::Rcout << "  sigma_mu: " ;
    for(size_t k = 0; k < q; k++) Rcpp::Rcout << " " << tree_pi[k].sigma_mu;
    Rcpp::Rcout << std::endl;
    
    Rcpp::Rcout << "y_col_mean: ";
    for(size_t k = 0; k < q; k++) Rcpp::Rcout << " " << y_col_mean[k];
    Rcpp::Rcout << endl;
    
    Rcpp::Rcout << "y_col_sd: ";
    for(size_t k = 0; k < q; k++) Rcpp::Rcout << " " << y_col_sd[k];
    Rcpp::Rcout << endl;
    
  }
  
  // Create data_info
  data_info di;
  di.n = n_obs;
  di.p = p;
  di.q = q;
  di.x = &x_ptr[0];
  di.y = &y_ptr[0];
  di.r_f = &r_full[0];
  di.delta = &delta_ptr[0];
  di.weight = weight;
  
  data_info dip;
  dip.n = n_pred;
  dip.p = p;
  dip.q = q;
  dip.x = &x_pred_ptr[0];
  
  if(verbose == true) Rcpp::Rcout << "  Initialized prior information classes" << endl;
  
  // create containers for output
  arma::cube f_train_samples = arma::zeros<arma::cube>(n_obs, q, nd);
  arma::cube f_test_samples = arma::zeros<arma::cube>(n_pred, q, nd);
  arma::mat sigma_samples = arma::zeros<arma::mat>(q,nd);
  
  // diagnostic information
  arma::mat alpha_samples = arma::zeros<arma::mat>(m*q, nd + burn);
  arma::mat tree_size_samples = arma::zeros<arma::mat>(m*q, nd + burn);
  if(verbose == true) Rcpp::Rcout << "  Starting MCMC" << std::endl;
  time_t tp;
  int time1 = time(&tp);

  for(int iter = 0; iter < (nd + burn); iter++){
    if(verbose == true){
      if(iter < burn & iter%50 == 0) Rcpp::Rcout << "  MCMC Iteration: " << iter << " of " << nd + burn << "; Burn-in" << std::endl;
      else if( (iter > burn & iter%50 == 0) || (iter == burn)) Rcpp::Rcout << "  MCMC Iteration: " << iter << " of " << nd + burn << "; Sampling" <<std::endl;
    }
    //Rcpp::Rcout << "iter = " << iter << endl;
    
    // Loop over each task and update the corresponding trees
    for(size_t k = 0; k < q; k++){
      //Rcpp::Rcout << "  k = " << k << endl;
      if(sigma[k] != sigma[k]) Rcpp::Rcout << "iter " << iter << " k = " << k << " sigma[k] is nan" << endl;
      for(size_t t = 0; t < m; t++){
        //Rcpp::Rcout << "  t = " << t;
        fit(t_vec[k][t], xi, di, ftemp); // get current fit of tree t for task k
        for(size_t i = 0; i < n_obs; i++){
          if(ftemp[i] != ftemp[i]) Rcpp::stop("nan in ftemp!");
          allfit[k + i*q] -= ftemp[i]; // subtract fit of tree t, task k from allfit
          if(delta_ptr[k + i*q] == 1) r_partial[i] = y_ptr[k + i*q] - allfit[k + i*q]; // update partial residual
        }
        
        
        
        //alpha_samples(t + k*m, iter) = bd_uni(t_vec[k][t], sigma[k], xi, di, tree_pi[k], gen); // do the birth/death move
        //drmu_uni(t_vec[k][t], sigma[k], xi, di, tree_pi[k], gen); // draw the new values of mu
        
        // 17 June 2019: need to pass the index of the task we are updating now
        alpha_samples(t + k*m, iter) = bd_uni(t_vec[k][t], sigma[k], xi, di, tree_pi[k], k, gen);
        //Rcpp::Rcout << " finished bd";
        drmu_uni(t_vec[k][t], sigma[k], xi, di, tree_pi[k], k, gen);
        //Rcpp::Rcout << "  drew mu!" << endl;
        
        fit(t_vec[k][t], xi, di, ftemp); // Update the fit from tree t, task k
        
        for(size_t i = 0; i < n_obs; i++){
          if(ftemp[i] != ftemp[i]) Rcpp::stop("nan in ftemp");
          allfit[k + i*q] += ftemp[i]; // add fit of tree t, outcome k back to allfit
          if(delta_ptr[k +i*q] == 1) r_full[k + i*q] = y_ptr[k + i*q] - allfit[k + i*q]; // update full residual
        }
      } // closes loop over the trees
    } // closes loop over the tasks
    
    update_sigma(sigma, sigma_pi, di, gen);
    if(iter >= burn){
      for(size_t k = 0; k < q; k++){
      // save training fits and sigma
        for(size_t i = 0; i < n_obs; i++) f_train_samples(i,k,iter-burn) = y_col_mean[k] + y_col_sd[k] * allfit[k + i*q];
        sigma_samples(k,iter-burn) = y_col_sd[k] * sigma[k];
        
        // clear the appropriate elements in allfit_pred and ftemp_pred
        for(size_t i = 0; i < n_pred; i++){
          ftemp_pred[i] = 0.0;
          allfit_pred[k + i*q] = 0.0;
        }
        // loop over the trees for outcome k to to build up the complete fit
        for(size_t t = 0; t < m; t++){
          fit(t_vec[k][t], xi, dip, ftemp_pred); // get fit of tree t, outcome k for test inputs
          for(size_t i = 0; i < n_pred; i++) allfit_pred[k + i*q] += ftemp_pred[i];
        }
        // now we can save the samples for the test outputs
        for(size_t i = 0; i < n_pred; i++) f_test_samples(i,k,iter-burn) = y_col_mean[k] + y_col_sd[k] * allfit_pred[k + i*q];
      }// closes loop over the tasks
    } // closes if checking that iter >= burn and that we must save our samples
    //Rcpp::Rcout << "iter " << iter << " sigma:" ;
    //for(size_t k = 0; k < q; k++) Rcpp::Rcout << " " << sigma[k] * y_col_sd[k];
    //Rcpp::Rcout << endl;

  } // closes main MCMC loop
  if(verbose == true) Rcpp::Rcout << "  Finished MCMC" << endl;

  int time2 = time(&tp);
  if(verbose == true) Rcpp::Rcout << "time for MCMC " << time2 - time1;
  
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
  //results["y_col_mean"] = y_col_mean;
  //results["y_col_sd"] = y_col_sd;
  results["f_train_samples"] = f_train_samples;
  results["f_test_samples"] = f_test_samples;
  results["sigma_samples"] = sigma_samples;
  //results["alpha_samples"] = alpha_samples
  //results["tree_depth_samples"] = tree_depth_samples;
  results["time"] = time2 - time1;
  
  
  return(results);
  
}
