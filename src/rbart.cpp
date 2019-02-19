//
//  bart_samples.cpp
//  
//
//  Created by Sameer Deshpande on 2/19/19.
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
arma::mat rbart(arma::mat X_pred, // points at which we wish to evaluate the function
                 Rcpp::List xinfo_list,
                 int n_samples = 1, // how many draws from the BART prior do we want
                 int m = 200, double alpha = 0.95, double beta = 2, double sigma_mu = 1.0){
  
  // setup the random number generator
  RNGScope scope;
  RNG gen;
  
  size_t n_pred = X_pred.n_rows;
  size_t p = X_pred.n_cols;
  
  arma::mat fit_samples = arma::zeros<arma::mat>(n_samples, n_pred); // one column per test point
  
  pinfo pi;
  pi.pbd = 1.0; // probability of a birth/death move. Always 1 since we don't consider swap/change
  pi.pb = 0.5; // probability of birth, given that birth/death move happens
  pi.alpha = alpha;
  pi.beta = beta;
  pi.sigma_mu[0] = sigma_mu;
  // don't need to change sigma_hat, lambda, or nu, since these are related to residual variance terms

  // create point for x_pred
  double* x_pred_ptr = new double[n_pred*p];
  for(size_t i = 0; i < n_pred; i++){
    for(size_t j = 0; j < p; j++){
      x_pred_ptr[j + i*p] = X_pred(i,j);
    }
  }
  
  
  dinfo dip; //
  dip.n = n_pred;
  dip.p = p;
  dip.q = 1;
  dip.k = 0;
  dip.x = &x_pred_ptr[0];
  
  
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
  
  double* ftemp = new double[n_pred]; // temporary fit of tree
  double* allfit = new double[n_pred]; // overall fit of tree
  for(size_t i = 0; i < n_pred; i++){
    allfit[i] = 0.0;
    ftemp[i] = 0.0;
  }
  
  
  std::vector<tree> t_vec(m);
  tree::npv bnv;
  for(int r = 0; r < n_samples; r++){
    t_vec.clear();
    // reset allfit
    for(size_t i = 0; i < n_pred; i++){
      allfit[i] = 0.0;
    }
    
    
    for(size_t t = 0; t < m; t++){
      draw_tree(t_vec[t], xi, alpha, beta, gen);
      bnv.clear();
      t_vec[t].getbots(bnv); // save bottom nodes of t_vec[t]
      for(tree::npv::size_type l = 0; l != bnv.size(); l++){
        bnv[l]->setm(sigma_mu * gen.normal()); // draw from the mu prior
      } // closes loop over the bottom nodes
    
      // now get the fits
      // note that fit overwrites whatever was in ftemp so no need to reset it
      fit(t_vec[t], xi, dip, ftemp);
      // update the overall fit
      for(size_t i = 0; i < n_pred; i++){
        allfit[i] += ftemp[i];
      }
      // at this time, allfit contains the sum of the first t trees
    } // closes loop over the trees
    for(size_t i = 0; i < n_pred; i++){
      fit_samples(r,i) = allfit[i];
    }
  } // closes loop over the replicates
  
  
  return(fit_samples);

}
