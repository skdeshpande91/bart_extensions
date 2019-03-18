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
                 int burn = 250, int nd = 1000, int m = 200, double kappa = 3, double nu = 1, double var_prob = 0.9)
{
  // Random number generator, used in all draws.
  RNGScope scope;
  RNG gen;
  
  Rcout << "\n*****Into bart main\n";
  
  // Get parameters about our data
  size_t n_obs = X.n_rows;
  size_t n_pred = X_pred.n_rows;
  size_t q = Y.n_cols;
  size_t p = X.n_cols;
  
  size_t n = Y.size(); // Should be n_obs * q
  
  // always center the Y's
  arma::mat Y_orig = Y; // make a copy of Y. Probably never refer back to this again.
  
  arma::vec y_col_mean(q);
  arma::vec y_col_sd(q);
  
  for(int k = 0; k < q; k++){
    y_col_mean(k) = mean(Y.col(k));
    Y.col(k) -= y_col_mean(k);
    y_col_sd(k) = arma::stddev(Y.col(k));
    Y.col(k) /= y_col_sd(k);
  }
  
  Rcpp::Rcout << "Original column means of Y:" ;
  for(size_t k = 0; k < q; k++) Rcpp::Rcout << " " << y_col_mean(k);
  Rcpp::Rcout << endl;
  
  Rcpp::Rcout << "Original column sd of Y:";
  for(size_t k = 0; k < q; k++) Rcpp::Rcout <<" " << y_col_sd(k);
  Rcpp::Rcout << endl;
  
  // create a double (pointer) for both x and y
  double* y_ptr = new double[n];
  double* x_ptr = new double[n_obs * p];
  double* x_pred_ptr = new double[n_pred * p];
  
  for(size_t i = 0; i < n_obs; i++){
    for(size_t k = 0; k < q; k++){
      //y_ptr[i + n_obs*k] = Y(i,k);
      y_ptr[k + i*q] = Y(i,k); //
    }
    for(size_t j = 0; j < p; j++){
      //x_ptr[i + n_obs*j] = X(i,j);
      x_ptr[j + i*p] = X(i,j);
    }
  }
  for(size_t i = 0; i < n_pred; i++){
    for(size_t j = 0; j < p; j++){
      //x_pred_ptr[i + n_pred*j] = X_pred(i,j);
      x_pred_ptr[j + i*p] = X_pred(i,j);
    }
  }
  Rcout << "created y_ptr, x_ptr, x_pred_ptr" << endl;
  
  // Read-in and format the cut-points
  xinfo xi;
  xi.resize(p);
  for(size_t j = 0; j < p; j++){
    NumericVector tmp = xinfo_list[j];
    std::vector<double> tmp2;
    for(int jj = 0; jj < tmp.size(); jj++){
      tmp2.push_back(tmp[jj]);
    }
    xi[j] = tmp2;
  }
  Rcout << "Created xi" << endl;
  
  // Inititalize Omega. For sep_bartFit Omega is always a diagonal matrix
  arma::mat Omega = zeros<mat>(q,q);
  Omega.eye();
  arma::mat Sigma = zeros<mat>(q,q);
  Sigma.eye();
  // Inititalize the S = R'R, where R is the residual matrix.
  arma::mat S = zeros<mat>(q,q);
  //double s_kk = 0.0; // will hold the diagonal elements of S when we need them

  // Set up the prior hyper-parameters
  pinfo pi;
  pi.pbd = 1.0; // prob of a birth/death move
  pi.pb = 0.5; // prob of birth move given birth/death move
  pi.alpha = 0.95; // prior probability that bottom node splies is alpha/(1 + d)^beta, d is depth
  pi.beta = 2.0; // 2 for bart means it is harder to build big trees
  pi.nu = nu;
  pi.sigma_mu.clear();
  pi.sigma_mu.reserve(q);
  
  pi.sigma_hat.clear();
  pi.sigma_hat.reserve(q);
  
  pi.lambda.clear();
  pi.lambda.reserve(q);
  // if p < n, we can set sigma_hat(k) to be the RMSE from fitting a linear model of Y.col(k) onto X
  // For now, we just set it equal to 1 (this is the wbart default for p > n)
  // Set-up the residual variance prior
  double chisq_quantile = 0.0;
  Function qchisq("qchisq");
  NumericVector tmp_quantile = qchisq(Named("p") = 1.0 - var_prob, Named("df") = nu);
  chisq_quantile = tmp_quantile[0];
  for(int k = 0; k < q; k++){
    pi.sigma_hat[k] = stddev(Y.col(k));
    pi.lambda[k] = (pi.sigma_hat[k] * pi.sigma_hat[k] * chisq_quantile)/nu;
    pi.sigma_mu[k] = ( (Y.col(k).max() - Y.col(k).min())/(2.0 * kappa * sqrt( (double) m))); // This is the default in wbart
  }
  
  Rcpp::Rcout << "  pi.sigma_hat:" ;
  for(size_t k = 0; k < q; k++) Rcpp::Rcout << " " << pi.sigma_hat[k] ;
  Rcpp::Rcout << endl;
  
  Rcpp::Rcout << "  pi.lambda:" ;
  for(size_t k = 0; k < q; k++) Rcpp::Rcout<< " " << pi.lambda[k];
  Rcpp::Rcout << endl;
  
  Rcpp::Rcout << "  pi.sigma_mu:" ;
  for(size_t k = 0; k < q; k++) Rcpp::Rcout << " " << pi.sigma_mu[k];
  Rcpp::Rcout << endl;
  
  
  

  
  // trees, data, and residuals
  std::vector<std::vector<tree> > t_vec(q, std::vector<tree>(m));
  
  double* allfit = new double[n];
  double* allfit_pred = new double[n_pred * q];
  double* r_full = new double[n]; // r_full[k + i*q] is full residual of observation i, outcome k
  double* r_partial = new double[n_obs]; // partial residual for observation i is r_partial[i]
  double* ftemp = new double[n_obs]; // temporarily holds current fit of a specific tree
  double* ftemp_pred = new double[n_pred];
  
  double ybar = 0.0;
  for(size_t k = 0; k < q; k++){
    ybar = arma::mean(Y.col(k));
    for(size_t t = 0; t < m; t++){
      t_vec[k][t].setm( ybar/ ( (double) m));
    }
    for(size_t i = 0; i < n_obs; i++){
      allfit[k + i*q] = ybar;
      r_full[k + i*q] = y_ptr[k + i*q] - allfit[k + i*q];
    }
  }
  for(size_t i = 0; i < n_obs; i++){
    r_partial[i] = 0.0;
    ftemp[i] = 0.0;
  }
  
  dinfo di;
  di.n = n_obs;
  di.p = p;
  di.q = q;
  di.k = 0;
  di.x = &x_ptr[0];
  di.r_f = &r_full[0];
  di.r_p = &r_partial[0];
  
  dinfo dip;
  dip.n = n_pred;
  dip.p = p;
  dip.q = q;
  dip.k = 0;
  dip.x = &x_pred_ptr[0];
  
  Rcpp::Rcout << "initialized pointers for residuals" << endl;

  // Samples
  arma::cube train_samples = zeros<cube>(n_obs, q, nd); // train_samples(i,k,iter) = allfit[k + i*q] * y_col_sd(k) + y_col_mean(k);
  arma::cube test_samples = zeros<cube>(n_pred, q, nd); // test_samples(i,k,iter) = allfit_pred[k+i*q] * y_col_sd(k) + y_col_mean(k)
  //arma::cube std_Omega_samples = zeros<cube>(q,q,nd); // Omega samples on the standardized scale
  arma::cube Omega_samples = zeros<cube>(q,q,nd); // Omega_samples(k,kk,iter) = std_Omega_samples(k,kk,iter)/y_col_sd(k)*y_col_sd(kk)
  arma::cube Sigma_samples = zeros<cube>(q,q,nd);
  // should we also compute Sigma samples?
  
  
  // diagnostics
  arma::cube S_samples = zeros<cube>(q,q,nd+burn); // save S = R'R for each iteration. Hope this is not too close to 0
  arma::mat alpha_samples = zeros<mat>(nd+burn, m*q); // alpha_samples(iter,t+m*k) is alpha for iteration iter,tree t, outcome k
  arma::mat tree_size_samples = zeros<mat>(nd+burn, m*q); // tree_depth_samples(iter,t+m*k) depth of tree t, outcome k for iteration iter
  
  // do MCMC here
  // inside the MCMC loop we can do the re-scaling and re-centering of the samples
  
  Rcpp::Rcout << "Starting MCMC" << endl;
  time_t tp;
  int time1 = time(&tp);
  for(int iter = 0; iter < (nd + burn); iter++){
    if(iter < burn & iter%50 == 0) Rcpp::Rcout << "  MCMC Iteration: " << iter << " of " << nd + burn << "; Burn-in" << endl;
    else if( (iter > burn & iter%50 == 0) || (iter == burn)) Rcpp::Rcout << "  MCMC Iteration: " << iter << " of " << nd + burn << "; Sampling" << endl;
    if(iter%100 == 0) Rcpp::checkUserInterrupt(); // check for interruption
    
    for(size_t k = 0; k < q; k++){
      di.k = k; // update k in di
      for(size_t t = 0; t < m; t++){
        fit(t_vec[k][t], xi, di, ftemp); // Fills in ftemp to hold current fitted values in tree t for outcome k
        for(size_t i = 0; i < n_obs; i++){
          if(ftemp[i] != ftemp[i]){
            Rcpp::Rcout << "outcome " << k << " tree " << t << " observation " << i << endl;
            Rcpp::stop("nan in ftemp");
          }
          allfit[k + i*di.q] = allfit[k + i*di.q] - ftemp[i]; // temporarily remove fit of tree t from allfit
          r_partial[i] = y_ptr[k + i*di.q] - allfit[k + i*di.q]; // compute r_partial
        } // closes loop over observations checking for nan in ftemp
        // now do the birth/death move
        alpha_samples(iter, t + m*k) = bd_multi(t_vec[k][t],Omega,xi, di,pi,gen);
        drmu_multi(t_vec[k][t], Omega, xi, di, pi, gen);
        
        // Update the fit
        fit(t_vec[k][t], xi, di, ftemp); // ftemp now has the new fitted values from tree t for outcome k
        for(size_t i = 0; i < n_obs; i++){
          if(ftemp[i] != ftemp[i]){
            Rcpp::Rcout << "outcome " << k << " tree " << t << " observation " << i << endl;
            Rcpp::stop("nan in ftemp");
          }
          allfit[k + i*di.q] += ftemp[i]; // allfit[k+i*di.q] previously had fit from (m-1) trees.
          r_full[k + i*di.q] = y_ptr[k + i*di.q] - allfit[k + i*di.q];
        }
        tree_size_samples(iter, t + m*k) = t_vec[k][t].treesize();
      } // closes loop over trees
    } // closes loop over outcomes
    
    // Now that trees have been updated it's time to update Omega
    // Let sigma_k be residal sd for outcome k (so omega_kk = 1/(sigma_k * sigma_k)
    // A priori sigma2_k ~ nu * lambda/chisq_nu = IG(nu/2, nu * lambda/2)
    // A posteriori sigma2_k ~ IG( (nu + n_obs)/2, (nu * lambda + S(k,k))/2) = (nu * lamba + S(k,k)) * IG( (nu + n_obs)/2, 1/2)
    // So to draw sigma2_k we just draw chisq_(nu + n_obs), invert it, and multipy it by (nu*lambda + S(k,k))
    // To draw omega we draw chisq(nu + n_obs) and divide it by (nu*lambda + S(k,k))
    
    S.zeros();
    for(size_t k = 0; k < q; k++){
      for(size_t kk = k; kk < q; kk++){
        for(int i = 0; i < n_obs; i++){
          S(k,kk) += r_full[k + i*q] * r_full[kk + i*q];
        }
        S(kk,k) = S(k,kk);
      }
      Omega(k,k) = gen.chi_square(pi.nu + di.n)/(S(k,k) + pi.nu * pi.lambda[k]);
    }
    S_samples.slice(iter) = S;
    
    
    
    // save the samples
    //[SKD] 16 January 2019: eventually add a thinning step here.
    // if(iter >= burn & (iter - 1) %% keep_every == 0)
    if(iter >= burn){
      for(size_t k = 0; k < q; k++){
        for(size_t i = 0; i < n_obs; i++) train_samples(i,k,iter-burn)= y_col_sd(k)*allfit[k + i*q] + y_col_mean(k); // save the fits for the training data
        for(size_t kk = k; kk < q; kk++){
          Omega_samples(k,kk,iter-burn) = Omega(k,kk)/(y_col_sd(k) * y_col_sd(kk));
          Omega_samples(kk,k,iter-burn) = Omega_samples(k,kk,iter-burn);
          Sigma_samples(k,kk,iter-burn) = y_col_sd(k) * y_col_sd(kk) * Sigma(k,kk);
          Sigma_samples(kk,k,iter-burn) = Sigma_samples(k,kk,iter-burn);
        }
      }
      // now get fits for testing data
      // first let us clear all of the values in allfit_pred
      for(size_t k = 0; k < q; k++){
        for(size_t i = 0; i < n_pred; i++) allfit_pred[k + i*q] = 0.0; // resets the values in allfit
      }
      for(size_t k = 0; k < q; k++){
        for(size_t t = 0; t < m; t++){
          fit(t_vec[k][t], xi, dip, ftemp_pred); // get the fit of tree t for outcome k
          for(size_t i = 0; i < n_pred; i++) allfit_pred[k + i*q] += ftemp_pred[i]; // update allfit_pred
        }
      }
      for(size_t k = 0; k < q; k++){
        for(size_t i = 0; i < n_pred; i++) test_samples(i,k,iter-burn) = y_col_sd(k) * allfit_pred[k + i*q] + y_col_mean(k);
      }
    } // closes if checking that iter > burn and that we should write the output
    
  } // closes main MCMC loop
  Rcpp::Rcout << "Finished MCMC" << endl;
  int time2 = time(&tp);
  Rcout << "time for MCMC: " << time2 - time1 << endl;

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
  
  List results;
  results["Y"] = Y;
  results["X"] = X;
  results["y_col_mean"] = y_col_mean;
  results["y_col_sd"] = y_col_sd;
  
  results["train_samples"] = train_samples;
  results["test_samples"] = test_samples;
  results["Omega_samples"] = Omega_samples;
  results["Sigma_samples"] = Sigma_samples;
  results["alpha_samples"] = alpha_samples;
  results["tree_size_samples"] = tree_size_samples;
  results["S_samples"] = S_samples;
  return(results);
  
}
