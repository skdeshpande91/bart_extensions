#ifndef GUARD_bd_h
#define GUARD_bd_h

#include<RcppArmadillo.h>
#include "rng.h"
#include "info.h"
#include "tree.h"
#include "funs.h"

//bool bd(tree& x, arma::mat const &Omega, xinfo& xi, dinfo& di, pinfo& pi, RNG& gen);
//double bd(tree &x, arma::mat const &Omega, xinfo &xi, dinfo &di, pinfo &pi, RNG &gen);

//double bd_multi(tree &x, const arma::mat &Omega, xinfo &xi, dinfo &di, pinfo &pi, RNG &gen);
double bd_uni(tree &x, const double &omega, xinfo &xi, dinfo &di, pinfo &pi, RNG &gen);


//overloaded to work with the new prior info classes
double bd_uni(tree &x, const double &sigma, xinfo &xi, data_info &di, tree_prior_info &tree_pi, RNG &gen);

// 17 June 2019: need to track the index of the task we're update
double bd_uni(tree &x, const double &sigma, xinfo &xi, data_info &di, tree_prior_info &tree_pi, size_t k,RNG &gen);


double bd_slfm(tree &x, const arma::mat &Phi, const arma::vec &sigma, xinfo &xi, dinfo_slfm &di, pinfo_slfm &pi, RNG &gen);
double bd_slfm(tree &x, const arma::mat &Phi, const std::vector<double> &sigma, xinfo &xi, data_info &di, tree_prior_info &tree_pi, RNG &gen); // overloaded for the new prior info classes


#endif
