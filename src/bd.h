#ifndef GUARD_bd_h
#define GUARD_bd_h

#include<RcppArmadillo.h>
#include "rng.h"
#include "info.h"
#include "tree.h"
#include "funs.h"

//bool bd(tree& x, arma::mat const &Omega, xinfo& xi, dinfo& di, pinfo& pi, RNG& gen);
//double bd(tree &x, arma::mat const &Omega, xinfo &xi, dinfo &di, pinfo &pi, RNG &gen);

double bd_multi(tree &x, const arma::mat &Omega, xinfo &xi, dinfo &di, pinfo &pi, RNG &gen);
double bd_uni(tree &x, const double &omega, xinfo &xi, dinfo &di, pinfo &pi, RNG &gen);

#endif
