//
//  tree_prior.h
//  
//
//  Created by Sameer Deshpande on 10/5/18.
//

#ifndef tree_prior_h
#define tree_prior_h

#include<RcppArmadillo.h>
#include "rng.h"
#include "info.h"
#include "tree.h"
#include "funs.h"

void draw_tree(tree& x, xinfo &xi, double alpha, double beta, RNG &gen);

#endif /* tree_prior_h */
