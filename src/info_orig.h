#ifndef GUARD_info_h
#define GUARD_info_h

#include <RcppArmadillo.h>
using namespace arma;

//============================================================
//data
//============================================================

class dinfo {
public:

   // Methods
  size_t p;  //number of vars
  size_t n;  //number of observations
  size_t q; // number of outcomes
  int k; // indexes the current outcome we are modifying
  double *x; // jth var of ith obs is *(x + p*i+j).  *x is first element of the data - is a pointer.
  double *r_p; // will temporarily hold the partial residual when fitting tree t of outcome k. length is n_obs
  double *r_f; // will hold full set of residuals
  // Constructor
  dinfo() {p=0;n=0;q=0;x=0;r_p=0; r_f = 0;}

};

class dinfo_slfm{
public:
  size_t p; // number of variables
  size_t n; // number of observations
  size_t q; // number of outcomes
  size_t D; // number of basis functions
  size_t d; // index of which basis function we are considering
  double *x; // j^th var of i^th obs is *(x + p*i+j)
  double *y; // i^th obs of task k is *(y +q*i + k)
  double *delta; // track the elements in delta_ptr
  double *af; // will track all of the elements in allfit
  double *uf; // will track all of the elements in ufit
  
  // constructor
  dinfo_slfm(){p=0;n=0;q=0;D = 0; d = 0;x=0;y=0;af=0;uf=0;}
};


//============================================================
//prior and mcmc
//============================================================

class pinfo
{
public:

   //----------------------------
   // Declare properties.
   //----------------------------
  //size_t q; // number of outcomes
  //mcmc info
  double pbd; //prob of birth/death
  double pb;  //prob of birth
  //prior info
  double alpha;
  double beta;
  
  std::vector<double> sigma_mu; // std deviation for the mu prior
  // This will be (Y.col(k).max() - Y.col(k).min())/(2 * sqrt( (double) m ) * kappa)
  // This is the default specification in wbart() from the BART package on CRAN
  
  // If p < n, we can take sigma_hat to be RMSE from a linear model
  // If p >= n, wbart() from the BART package on CRAN sets these to be stddev(Y.col(k))
  std::vector<double> sigma_hat; // initial over-estimtaes of the residual variance for each observation
  
  // In univariate BART, residual variance is sigma2 ~ nu*lambda/(chi^2_nu).
  // In multivariate BART, we will place an Inverse Wishart prior on Sigma. The centering matrix will be diagonal matrix Lambda
  // This corresponds to a Wishart prior on Omega with centering matrix Lambda^(-1).
  std::vector<double> lambda; 
  double nu;
  
  // constructor
  pinfo(){pbd = 1.0; pb = 0.5; alpha = 0.95; beta = 0.5; sigma_mu = std::vector<double>(1); sigma_hat = std::vector<double>(1); lambda = std::vector<double>(1); nu = 1.0;}
};


class pinfo_slfm
{
public:
  double pbd; // prob of birth/death
  double pb; // prob of birth
  double alpha;
  double beta;
  
  std::vector<double> sigma_mu; // std deviation of mu prior
  std::vector<double> sigma_phi; // std deviation of the phi prior
  std::vector<double> sigma_hat; // initial over-estimate of residual variances for each task
  std::vector<double> lambda; // scaling parameter for the prior on residuals variance
  double nu; // degrees of freedom for the prior on residuals variances
  
  double a_theta;
  double b_theta;
  
  // constructor
  pinfo_slfm(){pbd = 1.0; pb = 0.5; alpha = 0.95; beta = 0.5; sigma_mu = std::vector<double>(1); sigma_phi = std::vector<double>(1); sigma_hat = std::vector<double>(1); lambda = std::vector<double>(1); nu = 3.0; a_theta = 1.0; b_theta = 1.0;}
};

//============================================================
//sufficient statistics for 1 node
//============================================================

class sinfo
{
public:

  int n;
  std::vector<int> I; // holds the indices that are assigned to the node.
  // eventually we'll want to have a vector for holding value of omega.
  sinfo(){n = 0; I = std::vector<int>(1);}
};

#endif
