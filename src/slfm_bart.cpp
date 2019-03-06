//
//  slfm_bart.cpp
//  
//
//  Created by Sameer Deshpande on 2/20/19.
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

using namespace Rcpp;

// [[Rcpp::export]]
Rcpp::List slfm_bartFit(arma::mat Y,
                        arma::mat X,
                        arma::mat X_pred,
                        Rcpp::List xinfo_list,
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
  arma::mat Phi = arma::ones<arma::mat>(q,D); // Phi(k,d) tells us how much u_d contribute to f_k
  arma::vec sigma = arma::ones<arma::vec>(q); // sigma(k) is the residual SD for task k
  
  // set up prior hyper-parameters
  pinfo_slfm pi;
  pi.pbd = 1.0; // probability of a birth/death move
  pi.pb = 0.5; // probability of a birth move given birth/death move occurs
  pi.alpha = 0.95;
  pi.beta = 2.0;
  pi.sigma_mu.clear(); // prior sd of the mu parameters for each tree in the D basis functions
  pi.sigma_mu.reserve(D);
  
  pi.sigma_phi.clear(); // prior sd of the Phi parameters
  pi.sigma_phi.reserve(q); // we model the *rows* of Phi independently, each row has slightly different variance term to be consistent with data
  
  pi.sigma_hat.clear(); // over-estimate of the residual variance for each task. usually will just be 1.
  pi.sigma_hat.reserve(q);
  
  pi.lambda.clear(); // scaling value for the scaled-inverse chi-square prior on residual variances
  pi.lambda.reserve(q);
  pi.nu = nu; // df of the scaled-inverse chi-square prior on residual variances
  
  
  for(size_t d = 0; d < D; d++){
    pi.sigma_mu[d] = (y_col_max.max() - y_col_min.min())/(2.0 * kappa * sqrt( ((double) m) * ((double) D)));
  }
  
  // now set sigma_phi. This requires calling the qchisq function from R
  double chisq_quantile = 0.0;
  Function qchisq("qchisq");
  //NumericVector tmp_quantile = qchisq(Named("p") = 1.0 - var_prob, Named("df") = nu);
  NumericVector tmp_quantile = qchisq(Named("p") = var_prob, Named("df") = D);
  chisq_quantile = tmp_quantile[0];
  for(size_t k = 0; k < q; k++){
    pi.sigma_phi[k] = sqrt( ((double) D)/chisq_quantile ) * (y_col_max(k) - y_col_min(k))/(y_col_max.max() - y_col_min.min());
  }
  // now set sigma_hat and lambda
  tmp_quantile = qchisq(Named("p") = 1 - var_prob, Named("df") = nu);
  chisq_quantile = tmp_quantile[0];

  for(size_t k = 0; k < q; k++){
    pi.sigma_hat[k] = 1.0; // every column has variance 1
    pi.lambda[k] = chisq_quantile/nu;
  }
  Rcpp::Rcout << "Finished setting pi" << endl;
  
  Rcpp::Rcout << "  pi.sigma_mu:" ;
  for(size_t d = 0; d < D; d++) Rcpp::Rcout << " " << pi.sigma_mu[d] ;
  Rcpp::Rcout << endl;
  
  Rcpp::Rcout << "  pi.sigma_phi:";
  for(size_t k = 0; k < q; k++) Rcpp::Rcout << " " <<  pi.sigma_phi[k];
  Rcpp::Rcout << endl;
  
  Rcpp::Rcout << "  pi.sigma_hat:" ;
  for(size_t k = 0; k < q; k++) Rcpp::Rcout << " " << pi.sigma_hat[k] ;
  Rcpp::Rcout << endl;
  
  Rcpp::Rcout << "  pi.lambda:" ;
  for(size_t k = 0; k < q; k++) Rcpp::Rcout << " " << pi.lambda[k] ;
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
      allfit[k + i*q] = 0.0;
    }
    ftemp[i] = 0.0;
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
  dip.n = n_obs;
  dip.q = q;
  dip.d = 0;
  dip.x = &x_pred_ptr[0];
  
  Rcpp::Rcout << "  Created arrays to hold residuals, etc." << endl;
  
  // Remember we run the main BART procedure using centered and scaled responses
  // std_fit_samples holds fits for the centered and scaled data
  // fit_samples re-scales and re-centers back to the original scale
  arma::cube std_fit_samples = zeros<cube>(n_obs, q, nd);
  arma::cube fit_samples = zeros<cube>(n_obs, q, nd);
  arma::mat std_sigma_samples = arma::zeros<arma::mat>(nd,q); // for the residual sd on the standardized scale
  arma::mat sigma_samples = arma::zeros<arma::mat>(nd,q); // for the residual sd's on the original scale
  
  /*
   fit_samples(i,k,iter) = std_fit_samples(i,k,iter) * y_col_sd(k) + y_col_mean(k);
   sigma_samples(nd,iter) = std_sigma_samples(iter,k) * y_col_sd(k);
  */
  
  Rcpp::Rcout << "Starting MCMC" << endl;
  time_t tp;
  int time1 = time(&tp);
  for(int iter = 0; iter < (nd + burn); iter++){
    if(iter < burn & iter%50 == 0) Rcpp::Rcout << "  MCMC Iteration: " << iter << " of " << nd + burn << "; Burn-in" << endl;
    else if( (iter > burn & iter%50 == 0) || (iter == burn)) Rcpp::Rcout << "  MCMC Iteration: " << iter << " of " << nd + burn << "; Sampling" << endl;
    if(iter%100 == 0) Rcpp::checkUserInterrupt();
    
    // update the trees within each basis functions.

    for(size_t d = 0; d < D; d++){
      di.d = d; // in di, this lets us track which basis function we are updating
      for(size_t t = 0; t < m; t++){
        fit(t_vec[d][t], xi, di, ftemp); // fills in ftemp with current fitted values in tree t for basis function d
        for(size_t i = 0; i < n_obs; i++){
          if(ftemp[i] != ftemp[i]) Rcpp::stop("nan in ftemp!");
          ufit[d + i*D] -=ftemp[i]; // temporarily remove fit of tree t from overall fit of the basis function d
          for(size_t k = 0; k < q; k++){
            allfit[k + i*q] -= Phi(k,d) * ftemp[i]; // temporarily remove fit of tree t, basis function d from allfit
          }
        } // closes loop over observations
        // we are now ready to perform the birth/death move and then update the parameters
        
        bd_slfm(t_vec[d][t], Phi, sigma, xi, di, pi, gen);
        drmu_slfm(t_vec[d][t], Phi, sigma, xi, di, pi, gen);
        
        // now that we have new tree, we need to adjust ufit and allfit, since we had removed fit from this tree earlier
        fit(t_vec[d][t], xi, di, ftemp);
        for(size_t i = 0; i < n_obs; i++){
          if(ftemp[i] != ftemp[i]) Rcpp::stop("nan in ftemp!");
          ufit[d + i*D] += ftemp[i]; // add back fit of tree t to fit of basis function d
          for(size_t k = 0; k < q; k++){
            allfit[k + i*q] += Phi(k,d) * ftemp[i]; // add back fit of tree t, basis function d to allfit
          } // closes loop over tasks
        } // closes loop over obesrvations for updating allfit and ufit
      } // closes loop over the trees within basis function d
    } // closes loop over the basis functions
    
    // update the loadings Phi
    update_Phi_gaussian(Phi, sigma, di, pi, gen);

    // now that we have updated both the basis functions and the trees, we should update allfit
    for(size_t i = 0; i < n_obs; i++){
      for(size_t k = 0; k < q; k++){
        allfit[k + i*q] = 0.0;
        for(size_t d = 0; d < D; d++){
          allfit[k + i*q] += Phi(k,d) * ufit[d + i*D];
        } // closes loops over the basis functions
      } // closes loop over the tasks
    } // closes loop over observations
    
    // !! allfit has been updated
    
    // update the residual variances
    update_sigma(Phi, sigma, di, pi, gen);
    // Now collect all of the parameters
    if(iter >= burn){
      for(size_t k = 0; k < q; k++){
        for(size_t i = 0; i < n_obs; i++){
          std_fit_samples(i,k,iter-burn) = allfit[k + i*q];
          fit_samples(i,k,iter-burn) = y_col_sd(k) * allfit[k + i*q] + y_col_mean(k);
        } // closes loop over observations
        std_sigma_samples(iter-burn,k) = sigma(k);
        sigma_samples(iter-burn,k) = y_col_sd(k) * sigma(k);
      } // closes loop over tasks
    } // closes if checking that iter > burn and that we are saving samples
    
  } // closes main MCMC loop
  Rcpp::Rcout << "Finished MCMC" << endl;
  int time2 = time(&tp);
  Rcout << "time for MCMC: " << time2 - time1 << endl;
  
  
  delete[] y_ptr;
  delete[] delta_ptr;
  delete[] x_ptr;
  delete[] x_pred_ptr;

  Rcpp::List results;
  results["Y_orig"] = Y_orig;
  results["Y"] = Y;
  results["y_col_max"] = y_col_max;
  results["y_col_min"] = y_col_min;
  results["y_col_mean"] = y_col_mean;
  results["y_col_sd"] = y_col_sd;
  results["fit_samples"] = fit_samples;
  results["sigma_samples"] = sigma_samples;
  return(results);
}



/*
 n = n_obs * q
 double* allfit = new double[n]; // allfit[k + i * q] is fit of f_k(x_i)
 double* ftemp = new double[n_obs];
 double* ufit = new double[n]; // we actually will need ufit
 // possible to make this a matrix though. Contains the fits of
 
 
 // in back-fitting, we will make a call to fit, which updates ftemp
 // ftemp holds the fit of a single tree
 // then we update allfit as follows:
 for(int i = 0; i < n_obs; i++){
   for(int k = 0; k < q; k++){
     allfit[k + i*q] -= Phi(k,d) * ftemp[i]
   }
 }
 // then we make a call to bd_slfm, which takes different arguments than the other bd functions!
 // then we make a call to fit
 fit(t_vec[d][t], xi, di, ftemp)
 // now we need to update allfit, essentially adding back the fit from the tree we just updated
 for(int i = 0; i < n_obs; i++){
   for(int k = 0; k < q; k++){
     allfit[k + i*q] += Phi(k,d) * ftemp[i]
   }
 }
 
 // Once we finish updating all of the trees. We need to keep track of the u_fit matrix
 // This is to aid in updating Phi
 // To this end,
 
 
 
 // note that in the back-fitting, we will let allfit actually be the partial fit using all but one tree.
 // in the first step, we remove fit of a single tree from allfit, and then after updating the tree, we add it back.
 
 
 //double* u_partial = new double[n_obs]; // holds fit of (m-1) trees in u_d.
 
 
 // partial residual for observation i, outcome k is: r_partial[k + i*q]
 partial residual for observation i

 
 
 // It may not be wise to actually save an array of partial residuals because we have missing values
 // instead, we can keep track of an array of partial fits, and then when it is time to estimate parameters/ do the back-fitting compute the residuals on the fly
 
 
 // it's time now to update tree t in function d
 
 // make a call to fit fit(t_vec[d][t], ..., ftemp);
 fit(t_vec[d][t], xi, di, ftemp);
 // update u_fits
 for(int i = 0; i < n_obs; i++){
 
   for(int k = 0; k < q; k++){
 
   }
 }
 
 
 // now ftemp holds fitted value of the tree
 // u_fits[d + i*D] = u_fits[d + i*D] - ftemp[i] // u_partial now
 
 // once we update ftemp, we can go ahead
 
 // technically we could just update u_fits (in the same way we used to update u
 for(int i = 0; i < n; i++){
   r_partial[i] = y_ptr[i] -
 
 }
 
 
 for(int k = 0; k < q; k++){
   for(int i = 0; i < n_obs; i++){
     r_partial[i] =
   }
 }
 
 
*/
