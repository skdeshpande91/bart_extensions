#ifndef GUARD_funs_h
#define GUARD_funs_h

#include <RcppArmadillo.h>
#include <cmath>
#include <iostream>
#include "tree.h"
#include "info.h"
#include "rng.h"
//#include "GIGrvg.h"

using namespace arma;

// log sum exponential function

inline double logsumexp(const double &a, const double &b){
  return a < b ? b + log(1.0 + exp(a - b)) : a + log(1.0 + exp(b - a));
};

using std::cout;
using std::endl;

//pi and log(2*pi)
//#define PI 3.1415926535897931
#define LTPI 1.83787706640934536


typedef std::vector<std::vector<int> > lookup_t;

lookup_t make_lookup(Rcpp::IntegerMatrix lookup_table, Rcpp::IntegerVector cx);
void impute_x(int v, //variable index
              std::vector<int>& mask,
              int n, xinfo& xi, std::vector<double>& x, std::vector<vector<int> >& x_cat,
              std::vector<int>& cx, std::vector<int>& offsets, std::vector<int>& x_type,
              std::vector<tree>& t, std::vector<double>& y, double& sigma, RNG& rng);



//--------------------------------------------------
// MVN posterior utility function.
//Rcpp::List mvn_post_util(double sigma, vec mu0, mat Sigma0, vec n_vec, vec ybar_vec);

//normal density
double pn(
          double x,    //variate
          double m,    //mean
          double v     //variance
);
//--------------------------------------------------
//draw from a discrete distribution
int rdisc(
          double *p,   //vector of probabilities
          RNG& gen     //random number generator
);
//--------------------------------------------------
//evaluate tree tr on grid xi, write to os
void grm(tree& tr, xinfo& xi, std::ostream& os);
//--------------------------------------------------
//does a (bottom) node have variables you can split on?
bool cansplit(tree::tree_p n, xinfo& xi);
//--------------------------------------------------
//compute prob of a birth, goodbots will contain all the good bottom nodes
double getpb(tree &t, xinfo &xi, pinfo &pi, tree::npv &goodbots);
double getpb(tree &t, xinfo &xi, pinfo_slfm &pi, tree::npv &goodbots); // overloaded for SLFM
double getpb(tree &t, xinfo &xi, tree_prior_info &tree_pi, tree::npv &goodbots); // overloaded for new prior info classes
//--------------------------------------------------
//find variables n can split on, put their indices in goodvars
void getgoodvars(tree::tree_p n, xinfo& xi, std::vector<size_t>& goodvars);
//--------------------------------------------------
//get prob a node grows, 0 if no good vars, else a/(1+d)^b
double pgrow(tree::tree_p n, xinfo &xi, pinfo &pi);
double pgrow(tree::tree_p n, xinfo &xi, pinfo_slfm &pi);// overloaded for SLFM
double pgrow(tree::tree_p n, xinfo &xi, tree_prior_info &pi); // overloaded for new prior info classes

//--------------------------------------------------
// prepare the data
void prepare_y(arma::mat &Y, arma::vec &y_col_mean, arma::vec &y_col_sd, arma::vec &y_col_max, arma::vec &y_col_min); // for old versions
void prepare_y(arma::mat &Y, std::vector<double> &y_col_mean, std::vector<double> &y_col_sd, std::vector<double> &y_col_max, std::vector<double> &y_col_min);
void prepare_y(arma::vec &Y, double &y_mean, double &y_sd, double &y_max, double &y_min);

//--------------------------------------------------
// make cutpoints

void make_cutpoints(xinfo &xi, const size_t n_obs, const size_t p, double *x_ptr, const size_t n_cutpoints);

//--------------------------------------------------
//get sufficients stats for all bottom nodes
void allsuff(tree& x, xinfo& xi, dinfo& di, tree::npv& bnv, std::vector<sinfo>& sv);
void allsuff(tree& x, xinfo& xi, dinfo_slfm& di, tree::npv& bnv, std::vector<sinfo>& sv); // overloaded for  SLFM
void allsuff(tree &x, xinfo &xi, data_info &di, tree::npv &bnv, std::vector<sinfo> &sv); // overloaded for new prior info classes
//void allsuffhet(int k, tree& x, xinfo& xi, dinfo& di, double* phi, tree::npv& bnv, std::vector<sinfo>& sv);
//--------------------------------------------------
//get counts for all bottom nodes
//std::vector<int> counts(int k, tree& x, xinfo& xi, dinfo& di);
std::vector<int> counts(tree& x, xinfo& xi, dinfo& di, tree::npv& bnv);
std::vector<int> counts(tree &x, xinfo &xi, dinfo_slfm &di, tree::npv &bnv); // overloaded for SLFM
std::vector<int> counts(tree &x, xinfo &xi, data_info &di, tree::npv &bnv); // overloaded for the new prior info classes

//--------------------------------------------------
//update counts (inc or dec) to reflect observation i
/*
// deprecated:
void update_counts(int i, std::vector<int> &cts, tree &x, xinfo &xi, dinfo &di, int sign);
void update_counts(int i, std::vector<int>& cts, tree& x, xinfo& xi, dinfo_slfm &di, int sign); //overloaded for SLFM
void update_counts(int i, std::vector<int> &cts, tree &x, xinfo &xi, data_info &di, int sign); // overloaded for the new prior info classes
*/

void update_counts(int i, std::vector<int> &cts, tree &x, xinfo &xi, dinfo &di, tree::npv &bnv, int sign);
void update_counts(int i, std::vector<int>& cts, tree& x, xinfo& xi, dinfo_slfm &di, tree::npv& bnv, int sign); //overloaded for SLFM
void update_counts(int i, std::vector<int> &cts, tree &x, xinfo &xi, data_info &di, tree::npv &bnv, int sign); // overloaded the new prior info classes


void update_counts(int i, std::vector<int> &cts, tree &x, xinfo &xi, dinfo &di, std::map<tree::tree_cp,size_t>& bnmap, int sign);
void update_counts(int i, std::vector<int>& cts, tree& x, xinfo& xi, dinfo_slfm &di, std::map<tree::tree_cp,size_t>& bnmap, int sign); // overloaded version for SLFM
void update_counts(int i, std::vector<int> &cts, tree &x, xinfo &xi, data_info &di, std::map<tree::tree_cp,size_t>& bnmap, int sign); // overloaded for the new prior info classes


void update_counts(int i, std::vector<int> &cts, tree &x, xinfo &xi, dinfo &di, std::map<tree::tree_cp,size_t>& bnmap, int sign, tree::tree_cp &tbn);
void update_counts(int i, std::vector<int>& cts, tree& x, xinfo& xi, dinfo_slfm &di, std::map<tree::tree_cp,size_t>& bnmap, int sign, tree::tree_cp &tbn); // overloaded version for SLFM
void update_counts(int i, std::vector<int> &cts, tree &x, xinfo &xi, data_info &di, std::map<tree::tree_cp,size_t>& bnmap, int sign, tree::tree_cp &tbn); // overloaded for the new prior info classes



//--------------------------------------------------
//check minimum leaf size
bool min_leaf(int minct, std::vector<tree> &t, xinfo &xi, dinfo &di);
bool min_leaf(int minct, std::vector<tree> &t, xinfo &xi, dinfo_slfm &di); // overloaded for SLFM
bool min_leaf(int minct, std::vector<tree> &t, xinfo &xi, data_info &di); // overloaded for the new prior info classes
//--------------------------------------------------
//get sufficient stats for children (v,c) of node nx in tree x
//[SKD]: used in the birth proposals
void getsuff(tree &x, tree::tree_cp nx, size_t v, size_t c, xinfo &xi, dinfo &di, sinfo &sl, sinfo &sr);
void getsuff(tree &x, tree::tree_cp nx, size_t v, size_t c, xinfo &xi, dinfo_slfm &di, sinfo &sl, sinfo &sr); // overloaded for the SLFM
void getsuff(tree &x, tree::tree_cp nx, size_t v, size_t c, xinfo &xi, data_info &di, sinfo &sl, sinfo &sr); // overloaded for the new prior info classes

//--------------------------------------------------
//get sufficient stats for pair of bottom children nl(left) and nr(right) in tree x
//[SKD]: used in the death proposals
void getsuff(tree &x, tree::tree_cp nl, tree::tree_cp nr, xinfo &xi, dinfo &di, sinfo &sl, sinfo &sr);
void getsuff(tree &x, tree::tree_cp nl, tree::tree_cp nr, xinfo &xi, dinfo_slfm &di, sinfo &sl, sinfo &sr); // for SLFM
void getsuff(tree &x, tree::tree_cp nl, tree::tree_cp nr, xinfo &xi, data_info &di, sinfo &sl, sinfo &sr); //overloaded for the new prior info classes
//void getsuffhet(tree& x, tree::tree_cp nl, tree::tree_cp nr, xinfo& xi, dinfo& di, double* phi, sinfo& sl, sinfo& sr);

//--------------------------------------------------
// posterior mean and variance of mu
// these values are also used compute the log likelihood ratio

// for fully multivariate method we need to entire matrix Omega
//void mu_posterior_multi(double &mu_bar, double &V, const arma::mat &Omega, const sinfo &si, dinfo &di, const double sigma_mu);
void mu_posterior_uni(double &mu_bar, double &V, const double &omega, const sinfo &si, dinfo &di, const double sigma_mu);

void mu_posterior_uni(double &mu_bar, double &V, const double &sigma, const sinfo &si, data_info &di, tree_prior_info &tree_pi); // overloaded for the new prior info classes


void mu_posterior_slfm(double &mu_bar, double &V, const arma::mat Phi, const arma::vec sigma, sinfo &si, dinfo_slfm &di, double sigma_mu); //for SLFM

void mu_posterior_slfm(double &mu_bar, double &V, const arma::mat &Phi, const std::vector<double> &sigma, sinfo &si, data_info &di, tree_prior_info &tree_pi, phi_prior_info &phi_pi); // overloaded for the new prior info classes.


//--------------------------------------------------
//fit
void fit(tree& t, xinfo& xi, dinfo& di, std::vector<double>& fv);
//--------------------------------------------------
//fit
void fit(tree& t, xinfo& xi, dinfo& di, double* fv);
void fit(tree& t, xinfo& xi, dinfo_slfm& di, double* fv); // for SLFM
void fit(tree &t, xinfo &xi, data_info &di, double* fv); // overloaded for new prior info classes

template<class T>
double fit_i(T i, tree& t, xinfo& xi, dinfo& di)
{
  double *xx;
  double fv = 0.0;
  tree::tree_cp bn;
  xx = di.x + i*di.p;
  //for (size_t j=0; j<t.size(); ++j) {
  bn = t.bn(xx,xi);
  //fv = bn -> getm(di.t[i],di.tref);
  //fv = as_scalar(bn->getm(di.t[i]));

  //}
  return fv;
}

template<class T>
double fit_i(T i, std::vector<tree>& t, xinfo& xi, dinfo& di)
{
  double *xx;
  double fv = 0.0;
  tree::tree_cp bn;
  xx = di.x + i*di.p;
  for (size_t j=0; j<t.size(); ++j) {
		bn = t[j].bn(xx,xi);
		//fv += bn -> getm(di.t[i],di.tref);   // UPDATED, WAS (); in parens instead of (*di.t)
  }

  return fv;
}

template<class T>
double fit_i_mult(T i, std::vector<tree>& t, xinfo& xi, dinfo& di)
{
  double *xx;
  double fv = 1.0;
  tree::tree_cp bn;
  xx = di.x + i*di.p;
  for (size_t j=0; j<t.size(); ++j) {
  	bn = t[j].bn(xx,xi);
		//fv *= bn -> getm(di.t[i],di.tref);     // UPDATED, WAS (); in parens instead of (*di.t)
  }
  return fv;
}
//--------------------------------------------------
//partition
void partition(tree& t, xinfo& xi, dinfo& di, std::vector<size_t>& pv);
//--------------------------------------------------
// draw all the bottom node mu's

//void drmu(tree& t, arma::mat const &Omega, xinfo& xi, dinfo& di, pinfo& pi,RNG& gen);

//void drmu_multi(tree &t, const arma::mat  &Omega, xinfo &xi, dinfo &di, pinfo &pi, RNG &gen);
void drmu_uni(tree &t, const double &omega, xinfo &xi, dinfo &di, pinfo &pi, RNG &gen);
void drmu_uni(tree &t, const double &sigma, xinfo &xi, data_info &di, tree_prior_info &tree_pi, RNG &gen); // overloaded for the new prior info classes


void drmu_slfm(tree &t, const arma::mat Phi, const arma::vec sigma, xinfo &xi, dinfo_slfm &di, pinfo_slfm &pi, RNG &gen);
void drmu_slfm(tree &t, const arma::mat &Phi, const std::vector<double> &sigma, xinfo &xi, data_info &di, tree_prior_info &tree_pi, phi_prior_info &phi_pi, RNG &gen); // overloaded for the new prior info classes
//--------------------------------------------------
//write cutpoint information to screen
void prxi(xinfo& xi);
//--------------------------------------------------
//make xinfo = cutpoints
void makexinfo(size_t p, size_t n, double *x, xinfo& xi, size_t nc);
//get min/max for p predictors needed to make cutpoints.
void makeminmax(size_t p, size_t n, double *x, std::vector<double> &minx, std::vector<double> &maxx);
//make xinfo = cutpoints given minx/maxx vectors
void makexinfominmax(size_t p, xinfo& xi, size_t nc, std::vector<double> &minx, std::vector<double> &maxx);


void update_sigma_uni(double &sigma, sigma_prior_info &sigma_pi, data_info &di, RNG &gen);

// function to update sigma in multi-output setting
void update_sigma(std::vector<double> &sigma, std::vector<sigma_prior_info> &sigma_pi, data_info &di, RNG &gen);

//void update_latent_confounder(xinfo &xi, arma::cube &lat_conf_samples, arma::mat lat_conf_alpha_samples, iter, data_info &di, RNG &gen);

// Functions to update Phi in SLFM
void update_Phi_gaussian(arma::mat &Phi, const arma::vec &sigma, dinfo_slfm &di, pinfo_slfm &pi, RNG &gen);

// overloaded for the new prior information classes
void update_Phi_gaussian(arma::mat &Phi, const std::vector<double> &sigma, data_info &di, phi_prior_info &phi_pi, RNG &gen);


void update_Phi_ss(arma::mat &Phi, arma::vec &theta, const arma::vec &sigma, dinfo_slfm &di, pinfo_slfm &pi,  RNG &gen);
// Function to update sigma in SLFM
void update_sigma(const arma::mat &Phi, arma::vec &sigma, dinfo_slfm &di, pinfo_slfm &pi, RNG &gen);


//--------------------------------------------------
// Check if a vector is sorted.  For checking z and zpred for causal funbart.
bool is_sort(arma::vec x);

#endif
