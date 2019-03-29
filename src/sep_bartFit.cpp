//
//  sep_bartFit.cpp
//  
//
//  Created by Sameer Deshpande on 11/18/18.
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
List sep_bartFit(arma::mat Y,
                 arma::mat X,
                 arma::mat X_pred,
                 List xinfo_list,
                 int burn = 250, int nd = 1000, int m = 200, double kappa = 2, double nu = 3, double var_prob = 0.9, bool verbose = false)
{
  // Random number generator, used in all draws.
  RNGScope scope;
  RNG gen;
  
  // Get parameters about our data
  size_t n_obs = X.n_rows;
  size_t n_pred = X_pred.n_rows;
  size_t q = Y.n_cols;
  size_t p = X.n_cols;
  size_t n = Y.size(); // should be n_obs * q
  int D = Y.n_cols; // we are fixing the number of basis functions to be D = q. One basis function per task, so Phi is the identity.
  
  //Y.print();
  arma::mat Y_orig = Y; // just a copy of the Y that is initially passed into the function
  // Center and scale Y
  arma::vec y_col_mean = arma::zeros<arma::vec>(q); // holds the original mean of each column of Y
  arma::vec y_col_sd = arma::zeros<arma::vec>(q); // holds the original sd of each column of Y
  arma::vec y_col_max = arma::zeros<arma::vec>(q); // holds the max of the scaled columns of Y
  arma::vec y_col_min = arma::zeros<arma::vec>(q); // holds the min of the scaled columns of Y
  
  prepare_y(Y, y_col_mean, y_col_sd, y_col_max, y_col_min);
  
  if(verbose == true) Rcpp::Rcout << "  Centered and scaled Y" << endl;
  
  // create arrays (i.e. pointers for X, Y, X_pred, and delta)
  double* y_ptr = new double[n];
  double* delta_ptr = new double[n];
  double* x_ptr = new double[n_obs * p];
  double* x_pred_ptr = new double[n_pred * p];
  //double* delta_pred_ptr = new double[n_pred*q]; // just set all of these equal to 1
  
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
  if(verbose == true) Rcpp::Rcout << "  Created pointers" << endl;
  
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
  if(verbose == true) Rcpp::Rcout << "  Created cut-points" << endl;
  
  
  // introduce Phi and sigma
  arma::mat Phi = arma::zeros<arma::mat>(q,D) ;
  Phi.eye(); // in sep_bartFit, Phi is always the identity matrix. There are q latent basis functions, one for each task
  arma::vec sigma = arma::ones<arma::vec>(q); // sigma(k) is the residual SD for task k
  
  // set up prior hyper-parameters
  pinfo_slfm pi;
  pi.pbd = 1.0; // probability of a birth/death move
  pi.pb = 0.5; // probability of a birth move given birth/death move occurs
  pi.alpha = 0.95;
  pi.beta = 2.0;
  pi.sigma_mu.clear(); // prior sd of the mu parameters for each tree in the D basis functions
  pi.sigma_mu.reserve(D);
  
  pi.sigma_phi.clear(); // prior sd of the Phi parameters .. in sep_bartFit this is unused
  pi.sigma_phi.reserve(q); // we model the *rows* of Phi independently, each row has slightly different variance term to be consistent with data
  
  pi.sigma_hat.clear(); // over-estimate of the residual variance for each task. usually will just be 1.
  pi.sigma_hat.reserve(q);
  
  pi.lambda.clear(); // scaling value for the scaled-inverse chi-square prior on residual variances
  pi.lambda.reserve(q);
  pi.nu = nu; // df of the scaled-inverse chi-square prior on residual variances
  
  double chisq_quantile = 0.0;
  Function qchisq("qchisq");
  NumericVector tmp_quantile = qchisq(Named("p") = 1.0 - var_prob, Named("df") = nu);
  chisq_quantile = tmp_quantile[0];
  
  for(size_t k = 0; k < q; k++){
    pi.sigma_hat[k] = 1.0; // all columns have variance 1
    pi.lambda[k] = (pi.sigma_hat[k] * pi.sigma_hat[k] * chisq_quantile)/nu;
  }
  for(size_t d = 0; d < D; d++) pi.sigma_mu[d] = (y_col_max.max() - y_col_min.min())/(2.0 * kappa * sqrt( (double) m));
  
  if(verbose == true) Rcpp::Rcout << "Finished setting pi" << endl;
  // Set up trees, data, and residuals
  std::vector<std::vector<tree> > t_vec(D, std::vector<tree>(m));
  
  double* allfit = new double[n]; // allfit[k + i*q] is estimated fit of i^th training observation, k^th task
  double* ufit = new double[n_obs*D]; // ufit[d + i*D] is estimated fit of i^th observation, d^th basis
  double* ftemp = new double[n_obs]; // temporariliy holds current fit of a single tree
  
  double* allfit_pred = new double[n_pred*q]; // allfit_pred[k + i*q] is estimated fit of i^th test observation, k^th task
  double* ufit_pred = new double[n_pred*D]; // ufit_pred[d + i*D] is estimated fit of i^th test observation, d^th basis function
  double* ftemp_pred = new double[n_pred]; // temporarily holds fit
  
  // note that all of the columns have been centered
  // will set the terminal node parameter of each tree to be 0 to begin
  for(size_t d = 0; d < D; d++){
    for(size_t t = 0; t < m; t++) t_vec[d][t].setm(0.0);
  }
  for(size_t i = 0; i < n_obs; i++){
    for(size_t d = 0; d < D; d++) ufit[d + i*D] = 0.0;
    for(size_t k = 0; k < q; k++) allfit[k + i*q] = 0.0;
    ftemp[i] = 0.0;
  }
  
  // initialize allfit_pred and ufit_pred
  for(size_t i = 0; i < n_pred; i++){
    ftemp_pred[i] = 0.0;
    for(size_t d = 0; d < D; d++) ufit_pred[d + i*D] = 0.0;
    for(size_t k = 0; k < q; k++) allfit_pred[k + i*q] = 0.0;
  }
  
  dinfo_slfm di;
  di.p = p;
  di.n = n_obs;
  di.q = q;
  di.D = D;
  di.d = 0;
  di.x = &x_ptr[0];
  di.y = &y_ptr[0];
  di.delta = &delta_ptr[0];
  di.af = &allfit[0];
  di.uf = &ufit[0];
  
  dinfo_slfm dip;
  dip.p = p;
  dip.n = n_pred;
  dip.q = q;
  dip.d = 0;
  dip.x = &x_pred_ptr[0];
  
  
  if(verbose == true) Rcpp::Rcout << "  Created arrays to hold residuals, etc." << endl;
  
  // Remember we run the main BART procedure using centered and scaled responses
  arma::cube f_train_samples = arma::zeros<arma::cube>(n_obs, q, nd); // f_train_samples(i,k,iter) = y_col_sd(k) * allfit[k + i*q] + y_col_mean(k)
  arma::cube f_test_samples = arma::zeros<arma::cube>(n_pred, q, nd); // f_test_samples(i,k,iter) = y_col_sd(k) * allfit_pred[k + i*q] + y_col_mean(k)
  arma::mat sigma_samples = arma::zeros<arma::mat>(q, nd);  //
  
  // note: since Phi is being fixed at the identity, there is a single basis function for each f. So u_train and u_test would be redundant with f_train and f_test.
  
  //arma::cube u_train_samples = arma::zeros<arma::cube>(n_obs, D, nd); // u_train_samples(i,d,iter) = ufit[d+i*D] ... this is on standardized scale
  //arma::cube u_test_samples = arma::zeros<arma::cube>(n_pred, D, nd); // u_test_samples(i,d,iter) = ufit_pre[d+i*D] ... this is on the standardized scale
  
  //arma::mat sigma_samples = arma::zeros<arma::mat>(q, nd); // for residual standard deviations. each row represents a task. sigma(k,iter) = y_col_sd(k) * sigma[k]
  //arma::cube Phi_samples = arma::zeros<arma::cube>(q, D, nd); // since we are keeping Phi fixed to the I_q, there is no need to track Phi_samples
  
  if(verbose == true) Rcpp::Rcout << "Starting MCMC" << endl;
  time_t tp;
  int time1 = time(&tp);
  for(int iter = 0; iter < (nd + burn); iter++){
    if(iter < burn & iter%50 == 0){
      if(verbose == true) Rcpp::Rcout << "  MCMC Iteration: " << iter << " of " << nd + burn << "; Burn-in" << endl;
    }
    else if( (iter > burn & iter%50 == 0) || (iter == burn)){
      if(verbose == true) Rcpp::Rcout << "  MCMC Iteration: " << iter << " of " << nd + burn << "; Sampling" << endl;
    }
    if(iter%100 == 0) Rcpp::checkUserInterrupt();
    
    // update the trees within each basis functions.
    
    for(size_t d = 0; d < D; d++){
      di.d = d; // in di, this lets us track which basis function we are updating
      for(size_t t = 0; t < m; t++){
        fit(t_vec[d][t], xi, di, ftemp); // fills in ftemp with current fitted values in tree t for basis function d
        for(size_t i = 0; i < n_obs; i++){
          if(ftemp[i] != ftemp[i]) Rcpp::stop("nan in ftemp!");
          ufit[d + i*D] -=ftemp[i]; // temporarily remove fit of tree t from overall fit of the basis function d
          for(size_t k = 0; k < q; k++) allfit[k + i*q] -= Phi(k,d) * ftemp[i]; // temporariliy removes fit of tree t, basis function d from allfit
        } // closes loop over observations
        
        bd_slfm(t_vec[d][t], Phi, sigma, xi, di, pi, gen);
        drmu_slfm(t_vec[d][t], Phi, sigma, xi, di, pi, gen);
        
        // now that we have new tree, we need to adjust ufit and allfit, since we had removed fit from this tree earlier
        fit(t_vec[d][t], xi, di, ftemp);
        for(size_t i = 0; i < n_obs; i++){
          if(ftemp[i] != ftemp[i]) Rcpp::stop("nan in ftemp!");
          ufit[d + i*D] += ftemp[i]; // add back fit of tree t to fit of basis function d
          for(size_t k = 0; k < q; k++) allfit[k + i*q] += Phi(k,d) * ftemp[i]; // add back fit of tree t, basis function d to allfit
        } // closes loop over obesrvations for updating allfit and ufit
      } // closes loop over the trees within basis function d
    } // closes loop over the basis functions
    
    // Since Phi is the identity, we do not have to update Phi
    //update_Phi_gaussian(Phi, sigma, di, pi, gen);
    
    // This step is somewhat redundant but to be safe, we'll do it anyway
    // This updates allfit after ufit and Phi have been updated.
    for(size_t i = 0; i < n_obs; i++){
      for(size_t k = 0; k < q; k++){
        allfit[k + i*q] = 0.0;
        for(size_t d = 0; d < D; d++) allfit[k + i*q] += Phi(k,d) * ufit[d + i*D];
      } // closes loop over the tasks
    } // closes loop over observations
    
    // !! allfit has been updated
    
    // update the residual variances
    update_sigma(Phi, sigma, di, pi, gen);
    
    // Now save the samples
    if(iter >= burn){
      
      // save training fit
      for(size_t i = 0; i < n_obs; i++){
        for(size_t k = 0; k < q; k++) f_train_samples(i,k,iter-burn) = y_col_sd(k) * allfit[k + i*q] + y_col_mean(k);
        //for(size_t d = 0; d < D; d++) u_train_samples(i,d,iter-burn) = ufit[d + i*D];
      }
      
      // save the Phi's and sigmas
      for(size_t k = 0; k < q; k++){
        //for(size_t d = 0; d < D; d++) Phi_samples(k,d,iter-burn) = Phi(k,d) * y_col_sd(k);
        sigma_samples(k,iter-burn) = y_col_sd(k) * sigma(k);
      }
      
      // save test output. start by clearing all of the elements in ufit_pred
      for(size_t d = 0; d < D; d++){
        for(size_t i = 0; i < n_pred; i++) ufit_pred[d + i*D] = 0.0; // reset ufit_pred
        for(size_t t= 0; t < m; t++){
          fit(t_vec[d][t], xi, dip, ftemp_pred); // get the fit of tree t for basis function d for the test data
          for(size_t i = 0; i < n_pred; i++) ufit_pred[d + i*D] += ftemp_pred[i]; // update the appropriate elements in ufit_pred
        }
      }
      for(size_t k = 0; k < q; k++){
        for(size_t i = 0; i < n_pred; i++){
          allfit_pred[k + i*q] = 0.0; // reset allfit_pred[k+i*q];
          for(size_t d = 0; d < D; d++) allfit_pred[k+i*q] += Phi(k,d) * ufit_pred[d + i*D];
        }
      }
      for(size_t i = 0; i < n_pred; i++){
        for(size_t k = 0; k < q; k++) f_test_samples(i,k,iter-burn) = y_col_sd(k) * allfit_pred[k+i*q] + y_col_mean(k);
        //for(size_t d = 0; d < D; d++) u_test_samples(i,d,iter-burn) = ufit_pred[d + i*D];
      }
    } // closes if checking that iter > burn and that we are saving samples
    
  } // closes main MCMC loop
  if(verbose == true) Rcpp::Rcout << "Finished MCMC" << endl;
  int time2 = time(&tp);
  if (verbose == true) Rcout << "time for MCMC: " << time2 - time1 << endl;
  
  delete[] y_ptr;
  delete[] delta_ptr;
  delete[] x_ptr;
  delete[] x_pred_ptr;
  
  delete[] allfit;
  delete[] ufit;
  delete[] allfit_pred;
  delete[] ufit_pred;
  delete[] ftemp;
  delete[] ftemp_pred;
  
  Rcpp::List results;
  results["Y_orig"] = Y_orig;
  results["Y"] = Y;
  results["y_col_max"] = y_col_max;
  results["y_col_min"] = y_col_min;
  results["y_col_mean"] = y_col_mean;
  results["y_col_sd"] = y_col_sd;
  
  results["f_train_samples"] = f_train_samples;
  results["f_test_samples"] = f_test_samples;
  //results["u_train_samples"] = u_train_samples;
  //results["u_test_samples"] = u_test_samples;
  results["sigma_samples"] = sigma_samples;
  //results["Phi"] = Phi_samples;
  results["time"] = time2 - time1;
  
  return(results);
  
}
