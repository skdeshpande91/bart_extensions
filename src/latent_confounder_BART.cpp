//
//  latent_confounder_BART.cpp
//    Assume that there is a single latent confounder u_i ~ Bernoulli(0.5)
//    Parametrize E[y | X, Z, U] = f(X,Z,U) ~ BART
//    We can update u_i's with a Gibbs sampler.
//    Eventually we may want to use a continuous u and then use
//      Use a 0.95 * N(u, 2.38^2 * sigma_u^2) + 0.05 * N(u, 0.01^2) jump distribution
//      However, using a continuous u will require making some modifications to the tree prior/cutpoints. Unclear how to do that precisely
//  Created by Sameer Deshpande on 6 May 2019
//

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <vector>
#include <ctime>
#include <algorithm>

#include "rng.h"
#include "tree.h"
#include "info.h"
#include "bd.h"
#include "tree_prior.h"

#include <stdio.h>

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
Rcpp::List lat_conf_BART(arma::vec Y,
                         arma::mat X,
                         double u_prob = 0.5,
                         double weight = 1.0,
                         int burn = 500, int nd = 1000,
                         int m = 200, double kappa = 2,
                         double nu = 3, double var_prob = 0.9,
                         bool verbose = false)
{
  if(verbose == true) Rcpp::Rcout << "Entered lat_conf_BART" << endl;
  
  RNGScope scope;
  RNG gen;
  
  size_t n_obs = X.n_rows;
  size_t p_obs = X.n_cols; // number of measured confounders
  size_t p = X.n_cols + 1; // total number of measured + unmeasured confounders
  size_t q = 1; // this will always be 1
  
  size_t n = Y.size(); // always n_obs * q
  
  double y_mean = 0.0;
  double y_sd = 0.0;
  double y_max = 0.0;
  double y_min = 0.0;
  
  prepare_y(Y, y_mean, y_sd, y_max, y_min);
  
  // create pointer for x, y, and x_pred
  double* y_ptr = new double[n];
  double* delta_ptr = new double[n];
  double* x_ptr = new double[n_obs * p];
  
  double* x0_ptr = new double[n_obs * p]; // contains all of the observed X's and also U = 0
  double* x1_ptr = new double[n_obs * p]; // contains all of the observed X's and also U = 1
  
  for(size_t i = 0; i < n_obs; i++){
    if(Y(i) == Y(i)){ // true unles Y(i) is nan (i.e. Y_i missing)
      y_ptr[i] = Y(i);
      delta_ptr[i] = 1.0;
    }
    for(size_t j = 0; j < p_obs; j++){
      x_ptr[j + i * p] = X(i,j);
      x0_ptr[j + i * p] = X(i,j);
      x1_ptr[j + i * p] = X(i,j);
    }
    x0_ptr[p_obs + i * p] = 0.0;
    x1_ptr[p_obs + i * p] = 1.0;
    if(gen.uniform() < u_prob) x_ptr[p_obs + i * p] = 1.0;
    else x_ptr[p_obs + i * p] = 0.0;
  }
  
  // Create cut-points
  xinfo xi;
  make_cutpoints(xi, n_obs, p, x_ptr, 10000);
  
  Rcpp::Rcout << "Made cutpoints" << endl;
  
  double sigma = 1.0; // will track residual standard deviation
  
  // trees and pointers for fit and residuals
  std::vector<tree> t_vec(m);
  
  // posterior probability for u
  double prob_0 = 0.5;
  double prob_1 = 0.5;
  
  double* allfit = new double[n];
  double* allfit0 = new double[n];
  double* allfit1 = new double[n];
  double* r_full = new double[n];
  double* r_partial = new double[n_obs];
  
  double* ftemp = new double[n_obs];
  
  for(size_t t = 0; t < m; t++) t_vec[t].setm(0.0);
  for(size_t i = 0; i < n_obs; i++){
    allfit[i] = 0.0;
    allfit0[i] = 0.0;
    allfit1[i] = 0.0;
    if(delta_ptr[i] == 1) r_full[i] = y_ptr[i] - allfit[i];
  }
  for(size_t i = 0; i < n_obs; i++){
    r_partial[i] = 0.0;
    ftemp[i] = 0.0;
  }
  
  // Now create data_info, tree_prior_info, and sigma_prior_info objects
  
  //stuff for trees
  tree_prior_info  tree_pi;
  tree_pi.pbd = 1.0;
  tree_pi.pb = 0.5;
  tree_pi.alpha = 0.95;
  tree_pi.beta = 2.0;
  tree_pi.sigma_mu = (y_max - y_min)/(2.0 * kappa * sqrt( (double) m));
  tree_pi.r_p = &r_partial[0];
  
  sigma_prior_info sigma_pi;
  sigma_pi.sigma_hat = 1.0; // initial over-estimate of residual standard deviation
  double chisq_quantile = 0.0;
  Function qchisq("qchisq");
  NumericVector tmp_quantile = qchisq(Named("p") = 1.0 - var_prob, Named("df") = nu);
  chisq_quantile = tmp_quantile[0];
  sigma_pi.lambda = (sigma_pi.sigma_hat * sigma_pi.sigma_hat * chisq_quantile)/nu;
  sigma_pi.nu = nu;

  data_info di;
  di.n = n_obs;
  di.p = p;
  di.q = q;
  di.x = &x_ptr[0];
  di.y = &y_ptr[0];
  di.r_f = &r_full[0];
  di.delta = &delta_ptr[0];
  di.weight = weight;
  
  
  data_info di0;
  di0.n = n_obs;
  di0.p = p;
  di0.q = q;
  di0.x = &x0_ptr[0];
  di0.y = &y_ptr[0];
  di0.weight = weight;
  
  data_info di1;
  di1.n = n_obs;
  di1.p = p;
  di1.q = q;
  di1.x = &x_ptr[0];
  di1.y = &y_ptr[0];
  di1.weight = weight;
  
  // create containers for output
  arma::mat f_train_samples = arma::zeros<arma::mat>(n_obs, nd);
  arma::mat f0_test_samples = arma::zeros<arma::mat>(n_obs, nd); // holds predictions when u_i = 0
  arma::mat f1_test_samples = arma::zeros<arma::mat>(n_obs, nd); // holds predictions when u_i = 1
  
  arma::vec sigma_samples = arma::zeros<arma::vec>(nd);
  arma::mat lat_conf_samples = arma::zeros<arma::mat>(n_obs, nd);
  
  arma::mat alpha_samples = arma::zeros<arma::mat>(m, nd + burn);

  Rcpp::Rcout << "Starting MCMC" << endl;
  time_t tp;
  int time1 = time(&tp);
  for(size_t iter = 0; iter < (nd + burn); iter++){
    if(verbose == true){
      if(iter < burn & iter%50 == 0) Rcpp::Rcout << "  MCMC Iteration: " << iter << " of " << nd + burn << "; Burn-in" << endl;
      else if( (iter > burn & iter%50 == 0) || (iter == burn)) Rcpp::Rcout << "  MCMC Iteration: " << iter << " of " << nd + burn << "; Sampling" << endl;
    }
    if(iter%100 == 0) Rcpp::checkUserInterrupt(); // check for user interruption every 100 iterations
    // To update u conditional on all of the trees, we need to compute the fit when u = 0 and u = 1
    for(size_t i = 0; i < n; i++){
      allfit0[i] = 0.0;
      allfit1[i] = 0.0;
    }
    for(size_t t = 0; t < m; t++){
      fit(t_vec[t], xi, di0, ftemp);
      for(size_t i = 0; i < n; i++) allfit0[i] += ftemp[i];
      fit(t_vec[t], xi, di1, ftemp);
      for(size_t i = 0; i < n; i++) allfit1[i] += ftemp[i];
    }
    for(size_t i = 0; i < n_obs; i++){
      // probability that u_i = 1 is proportional to (2*pi*sigma*sigma)^(-weight/2) * exp(-weight/(2 * sigma * sigma) * (y_ptr[i] - f(x,1))^2) * u_prob
      prob_1 = exp(-0.5 * weight * (y_ptr[i] - allfit1[i]) * (y_ptr[i] - allfit1[i])) * u_prob;
      prob_0 = exp(-0.5 * weight * (y_ptr[i] - allfit0[i]) * (y_ptr[i] - allfit0[i])) * (1.0 - u_prob);
      
      if(gen.uniform() < prob_1/(prob_1 + prob_0)) x_ptr[p_obs + i * p] = 1.0;
      else x_ptr[p_obs + i*p] = 0.0;
    }
    // Updating trees
    for(size_t t = 0; t < m; t++){
      fit(t_vec[t], xi, di, ftemp); // Get the current fit of the tree
      for(size_t i = 0; i < n_obs; i++){
        if(ftemp[i] != ftemp[i]){
          Rcpp::Rcout << "tree " << t << " observation " << i << endl;
          Rcpp::stop("nan in ftemp");
        } // closes if checking whether ftemp[i] is nan
        allfit[i] = allfit[i] - ftemp[i]; // temporarily remove fit of tree t from allfit
        if(delta_ptr[i] == 1) r_partial[i] = y_ptr[i] - allfit[i]; // allfit contains fit of (m-1) trees so this is the correct value of r_partial
      } // closes loop over observations updating allfit and r_partial
      
      alpha_samples(t, iter) = bd_uni(t_vec[t], sigma, xi, di, tree_pi,gen); // do the birth/death move
      drmu_uni(t_vec[t], sigma, xi, di, tree_pi, gen); // Draw the new mu parameters
      fit(t_vec[t], xi, di, ftemp); // Update the fit
      
      for(size_t i = 0; i < n_obs; i++){
        if(ftemp[i] != ftemp[i]) Rcpp::stop("nan in ftemp");
        allfit[i] += ftemp[i]; // add fit of tree t back to allfit
        if(delta_ptr[i] == 1) r_full[i] = y_ptr[i] - allfit[i]; // update the full residual
      }
    } // closes loop over trees
    // Now we can update sigma
    update_sigma_uni(sigma, sigma_pi, di, gen);
    
    if(iter >= burn){
      for(size_t i = 0; i < n_obs; i++){
        lat_conf_samples(i, iter-burn) = x_ptr[p_obs + i*p];
        f_train_samples(i, iter-burn) = y_mean + y_sd * allfit[i];
        f0_test_samples(i, iter-burn) = y_mean + y_sd * allfit0[i];
        f1_test_samples(i, iter-burn) = y_mean + y_sd * allfit1[i];
      }
      sigma_samples(iter-burn) = y_sd * sigma;
      
    }
    
  } // closes main MCMC loop
  Rcpp::Rcout << "Finished MCMC" << endl;
  int time2 = time(&tp);
  Rcpp::Rcout << "time for MCMC: " << time2 - time1 << endl;
  
  
  Rcpp::List results;
  results["f_train_samples"] = f_train_samples;
  results["u_samples"] = lat_conf_samples;
  results["f1_test_samples"] = f1_test_samples;
  results["f0_test_samples"] = f0_test_samples;
  results["time"] = time2 - time1;

  return(results);
  
}
