//
//  slfm_BART_res.cpp
//  Uses the new prior information classes.
//  Allows for weighted likelihood
//  Re-parametrizes f_k(x) = h_k(x) + \sum{phi_{kd}u_d(x)}
//  Created by Sameer Deshpande on 6/17/19.
//

#include <stdio.h>
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

using namespace Rcpp;

// [[Rcpp::export]]
Rcpp::List slfm_BART_res(arma::mat Y,
                         arma::mat X,
                         arma::mat X_pred,
                         Rcpp::List xinfo_list,
                         double weight = 1.0,
                         int burn = 250, int nd = 1000,
                         size_t D = 10, size_t m_u = 50, size_t m_h = 100,
                         double kappa = 2.0,
                         double nu = 3, double var_prob = 0.9,
                         bool verbose = false)
{
  if(verbose == true) Rcpp::Rcout << "Entering slfm_BART_res" << endl;
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
  
  double y_min = y_col_min[0];
  double y_max = y_col_max[0];
  for(size_t k = 1; k < q; k++){
    if(y_col_min[k] < y_min) y_min = y_col_min[k];
    if(y_col_max[k] > y_max) y_max = y_col_max[k];
  }
  
  
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
  
  // Read & format the cutpoints
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
  
  // Initialize Phi
  arma::mat Phi = arma::ones<arma::mat>(q,D);
  
  // Initialize trees
  std::vector<std::vector<tree> > t_vec_u(D, std::vector<tree>(m_u)); // trees for shared basis functions
  std::vector<std::vector<tree> > t_vec_h(q, std::vector<tree>(m_h)); // trees for task-specific trees
  
  double* allfit = new double[n_obs * q];
  double* allfit_pred = new double[n_pred*q];
  double* ftemp = new double[n_obs];
  double* ftemp_pred = new double[n_pred];
  
  double* ufit = new double[n_obs * D]; // ufit[d + i*D] is u_d(x_i)
  double* ufit_pred = new double[n_pred * D];
  
  double* hfit = new double[n_obs * q]; // hfit[k + i*q] is fit h_k(x_i)
  //double* hfit_pred = new double[n_pred * q];
  
  double* r_full = new double[n_obs * q]; // holds full residuals for training observations
  
  // when we hold out a single tree from h_k, it affects only the responses for task k
  // when we hold out a single tree from u_d, it can affect every fitted value.
  // create a single pointer for the set of partial residuals and track that within
  // the tree prior class
  
// SORT OUT THE r_partial's for this parametrization!
  double* r_partial_h = new double[n_obs]; // for updating h_k, we only need n_obs
  double* r_partial_u = new double[n_obs * q]; // when updated u_d we need a lot more!
  
  // initialize the task-specific stuff
  for(size_t k = 0; k < q; k++){
    for(size_t t = 0; t < m_h; t++) t_vec_h[k][t].setm(0.0);
    for(size_t i = 0; i < n_obs; i++){
      hfit[k + i*q] = 0.0;
      allfit[k + i*q] = 0.0;
      if(delta_ptr[k + i*q] == 1){
        r_full[k + i*q] = y_ptr[k + i*q] - allfit[k + i*q];
        r_partial_u[k + i*q] = 0.0; // this will be re-written over and over again
      }
    }
    
    for(size_t i = 0; i < n_pred; i++) allfit_pred[k + i*q] = 0.0;
  }
  
  // initialize stuff for the shared basis elements
  for(size_t d = 0; d < D; d++){
    for(size_t t = 0; t < m_u; t++) t_vec_u[d][t].setm(0.0);
    for(size_t i = 0; i < n_obs; i++) ufit[d + i*D] = 0.0;
    for(size_t i = 0; i < n_pred; i++) ufit_pred[d + i*D] = 0.0;
  }
  
  for(size_t i = 0; i < n_obs; i++){
    ftemp[i] = 0.0;
    r_partial_h[i] = 0.0; // initial value doesn't really matter
  }
  for(size_t i = 0; i < n_pred; i++) ftemp_pred[i] = 0.0;
  
  
  // set the tree prior info
  
  // the u_d's are IID a priori. So it is enough to create a single tree_prior_info object for them
  tree_prior_info u_tree_pi;
  u_tree_pi.pbd = 1.0;
  u_tree_pi.pb = 0.5;
  u_tree_pi.alpha = 0.95;
  u_tree_pi.beta = 2.0;
  u_tree_pi.sigma_mu = 1/sqrt(2.0 * kappa * ( (double) D )* (double) m_u); // this is different!
  u_tree_pi.r_p = &r_partial_u[0]; // tracks partial residual
  
  std::vector<tree_prior_info> h_tree_pi(q);
  for(size_t k = 0; k < q; k++){
    h_tree_pi[k].pbd = 1.0;
    h_tree_pi[k].pb = 0.5;
    h_tree_pi[k].alpha = 0.95;
    h_tree_pi[k].beta = 2.0;
    h_tree_pi[k].sigma_mu = (y_col_max[k] - y_col_min[k])/(2.0 * kappa * sqrt((double) m_h));
    h_tree_pi[k].r_p = &r_partial_h[0];
  }
  
  // We have q residuals so we need a vector of sigma_prior_info objects
  std::vector<sigma_prior_info> sigma_pi(q);
  double chisq_quantile = 0.0;
  Function qchisq("qchisq"); // uses R's built-in quantile function
  Rcpp::NumericVector tmp_quantile = qchisq(Named("p") = 1.0 - var_prob, Named("df") = nu);
  chisq_quantile = tmp_quantile[0];
  for(size_t k = 0; k < q; k++){
    sigma_pi[k].sigma_hat = 1.0;
    sigma_pi[k].nu = nu;
    sigma_pi[k].lambda = chisq_quantile/nu;
  }
  
  // setup prior for Phi
  phi_prior_info phi_pi;
  phi_pi.D = D;
  phi_pi.q = q;
  phi_pi.d = 0; // this lets us keep track of which basis function is being updated at all times
  phi_pi.uf = &ufit[0];
  phi_pi.sigma_phi.clear();
  phi_pi.sigma_phi.resize(q);
  tmp_quantile = qchisq(Named("p") = var_prob, Named("df") = D);
  chisq_quantile = tmp_quantile[0];
  for(size_t k = 0; k < q; k++) phi_pi.sigma_phi[k] = sqrt( ((double) D)/chisq_quantile) * (y_col_max[k] - y_col_min[k])/(2.0 * kappa);
  
  // set up data info
  data_info di;
  di.n = n_obs;
  di.p = p;
  di.q = q;
  di.x = &x_ptr[0];
  di.y = &y_ptr[0];
  di.af = &allfit[0];
  di.r_f = &r_full[0];
  di.delta = &delta_ptr[0];
  di.weight = weight;
  
  data_info dip;
  dip.n = n_pred;
  dip.p = p;
  dip.q = q;
  dip.x = &x_pred_ptr[0];
  
  // create containers for output
  arma::cube f_train_samples = arma::zeros<arma::cube>(n_obs, q, nd);
  arma::cube f_test_samples = arma::zeros<arma::cube>(n_pred, q, nd);
  
  arma::mat sigma_samples = arma::zeros<arma::mat>(q, nd);
  
  double tmp_alpha; // temporary value to hold MH acceptance probability
  
  if(verbose == true){
    
    Rcpp::Rcout << "h sigma_mu : ";
    for(size_t k = 0; k < q; k++) Rcpp::Rcout << h_tree_pi[k].sigma_mu;
    Rcpp::Rcout << endl;
    
    Rcpp::Rcout << "u sigma_mu: " << u_tree_pi.sigma_mu << endl;
    Rcpp::Rcout << "sigma_phi: " ;
    for(size_t k = 0; k < q; k++) Rcpp::Rcout << " " << phi_pi.sigma_phi[k];
    Rcpp::Rcout << endl;
    
    Rcpp::Rcout << "lambda: ";
    for(size_t k = 0; k < q; k++) Rcpp::Rcout << " " << sigma_pi[k].lambda;
    Rcpp::Rcout << endl;
  }
  
  
  if(verbose == true) Rcpp::Rcout << "  Starting MCMC" << endl;
  time_t tp;
  int time1 = time(&tp);
  
  for(int iter = 0; iter < nd + burn; iter++){
    if(verbose == true){
      if(iter < burn & iter%50 == 0) Rcpp::Rcout << "  MCMC Iteration: " << iter << " of " << nd + burn << "; Burn-in" << std::endl;
      else if( (iter > burn & iter%50 == 0) || (iter == burn)) Rcpp::Rcout << "  MCMC Iteration: " << iter << " of " << nd + burn << "; Sampling" <<std::endl;
    }
    // Start by updating the individual task components h_k
    for(size_t k = 0; k < q; k++){
      for(size_t t = 0; t < m_h; t++){
        fit(t_vec_h[k][t], xi, di, ftemp);
        for(size_t i = 0; i < n_obs; i++){
          if(ftemp[i] != ftemp[i]) Rcpp::stop("nan in ftemp");
          hfit[k + i*q] -= ftemp[i];
          allfit[k + i*q] -= ftemp[i]; // allfit now is missing the fit of tree t in h_k
          if(delta_ptr[k + i*q] == 1) r_partial_h[i] = y_ptr[k + i*q] - allfit[k + i*q];
        } // closes loop over observations that removes fit
        tmp_alpha = bd_uni(t_vec_h[k][t], sigma[k], xi, di, h_tree_pi[k], k, gen);
        drmu_uni(t_vec_h[k][t], sigma[k], xi, di, h_tree_pi[k], k, gen);
        
        fit(t_vec_h[k][t], xi, di, ftemp);
        for(size_t i = 0; i < n_obs; i++){
          if(ftemp[i] != ftemp[i]) Rcpp::stop("nan in ftemp!");
          hfit[k + i*q] += ftemp[i];
          allfit[k + i*q] += ftemp[i];
        } // closes loop over observations that adds fit back in
      } // closes loop over trees for h_k
    } // closes loop over tasks
    
    // Now we update the shared basis elements u
    for(size_t d = 0; d < D; d++){
      phi_pi.d = d; // keep track of which basis function we are updating
      for(size_t t = 0; t < m_u; t++){
        fit(t_vec_u[d][t], xi, di, ftemp); // get fit of tree t from basis element u_d
        for(size_t i = 0; i < n_obs; i++){
          if(ftemp[i] != ftemp[i]) Rcpp::stop("nan in ftemp!");
          ufit[d + i*D] -= ftemp[i]; // temporarily remove fit of tree t, basis element u_d from ufit
          for(size_t k = 0; k < q; k++){
            allfit[k + i*q] -= Phi(k,d) * ftemp[i];
            if(delta_ptr[k + i*q] == 1) r_partial_u[k + i*q] = y_ptr[k + i*q] - allfit[k + i*q];
          }
        } // closes loop over observations that removes fit of tree t, basis element d from allfit
        
        tmp_alpha = bd_slfm(t_vec_u[d][t], Phi, sigma, xi, di, u_tree_pi, phi_pi, gen);
        drmu_slfm(t_vec_u[d][t], Phi, sigma, xi, di, u_tree_pi, phi_pi, gen);
        
        fit(t_vec_u[d][t], xi, di, ftemp);
        for(size_t i = 0; i < n_obs; i++){
          if(ftemp[i] != ftemp[i]) Rcpp::stop("nan in ftemp");
          ufit[d + i*D] += ftemp[i]; // add fit of tree t back to basis function u_d
          for(size_t k = 0; k < q; k++) allfit[k + i*q] += Phi(k,d) * ftemp[i];
        }
      } // closes loop over trees in basis element u_d
    } // closes loop over d
    
    // Update Phi
    update_Phi_gaussian(Phi, sigma, di, phi_pi, gen);
    
    // now that both Phi, the basis functions, and h_k's have been updated, we should re-compute a few things
    for(size_t i = 0; i < n_obs; i++){
      for(size_t k = 0; k < q; k++){
        allfit[k + i*q] = hfit[k + i*q]; // add in the task-specific function
        for(size_t d = 0; d < D; d++) allfit[k + i*q] += Phi(k,d) * ufit[d + i*D];
        if(delta_ptr[k + i*q] == 1) r_full[k + i*q] = y_ptr[k + i*q] - allfit[k + i*q];
      }
    }
    
    // Now we update sigma
    update_sigma(sigma, sigma_pi, di, gen);
    
    if(iter >= burn){
      
      // save sigma samples
      for(size_t k = 0; k < q; k++) sigma_samples(k, iter-burn) = y_col_sd[k] * sigma[k];
      
      
      //  training samples
      for(size_t k = 0; k < q; k++){
        for(size_t i = 0; i < n_obs; i++) f_train_samples(i, k, iter-burn) = y_col_mean[k] + y_col_sd[k] * allfit[k + i*q];
      }
      
      // testing samples: clear everything
      for(size_t i = 0; i < n_pred; i++){
        ftemp_pred[i] = 0.0;
        for(size_t d = 0; d < D; d++) ufit_pred[d + i*D] = 0.0;
        for(size_t k = 0; k < q; k++) allfit_pred[k + i*q] = 0.0;
      }
      
      // set allfit_pred[k + i*q] to be the value of h_k(x*_i)
      for(size_t k = 0; k < q; k++){
        for(size_t t = 0; t < m_h; t++){
          fit(t_vec_h[k][t], xi, dip, ftemp_pred); // get fit of tree t for h_k
          for(size_t i = 0; i < n_pred; i++) allfit_pred[k + i*q] += ftemp_pred[i];
        }
      }
      // compute u_d(x*_i) for each test point x*_i
      for(size_t d = 0; d < D; d++){
        for(size_t t = 0; t < m_u; t++){
          fit(t_vec_u[d][t], xi, dip, ftemp_pred); // get fit of tree t, basis element u_d
          for(size_t i = 0; i < n_pred; i++) ufit_pred[d + i*D] += ftemp_pred[i];
        }
      }
      
      // add Phi(k,d) u_d(x*_i) to allfit[k + i*q] and then save it
      for(size_t k = 0; k < q; k++){
        for(size_t i = 0; i < n_pred; i++){
          for(size_t d = 0; d < D; d++) allfit_pred[k + i*q] += Phi(k,d) * ufit_pred[d + i*D];
          f_test_samples(i, k, iter-burn) = y_col_mean[k] + y_col_sd[k] * allfit_pred[k + i*q];
        }
      }
    } // closes if checking that iter > burn and that we are saving samples
  } // closes main MCMC loop
  if(verbose == true) Rcpp::Rcout << "Finished MCMC" << endl;
  int time2 = time(&tp);
  if(verbose == true) Rcpp::Rcout << "time for MCMC : " << time2 - time1 << endl;
    
  delete[] y_ptr;
  delete[] delta_ptr;
  delete[] x_ptr;
  delete[] x_pred_ptr;
  
  delete[] allfit;
  delete[] allfit_pred;
  delete[] hfit;
  delete[] ufit;
  delete[] ufit_pred;
  delete[] r_full;
  delete[] r_partial_h;
  delete[] r_partial_u;
  delete[] ftemp;
  delete[] ftemp_pred;
  
  Rcpp::List results;
  results["f_train_samples"] = f_train_samples;
  results["f_test_samples"] = f_test_samples;
  results["sigma_samples"] = sigma_samples;
  results["time"] = time2 - time1;
  return(results);
}
