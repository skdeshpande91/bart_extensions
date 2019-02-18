//
//  tree_prior.cpp
//  
//
//  Created by Sameer Deshpande on 10/4/18.
//
#include<RcppArmadillo.h>

#include<iostream>
#include "rng.h"
#include "funs.h"
#include "info.h"
#include "tree.h"
#include "tree_prior.h"
#include <stdio.h>

using namespace Rcpp;
using namespace arma;
using namespace std;

void draw_tree(tree& x, xinfo &xi, double alpha, double beta, RNG &gen){
  tree::npv goodbots;
  tree::npv bnv; // all bottom nodes
  tree::tree_p nx;
  size_t dnx;
  //double PBx = 0.0; // won't end up using this but getpb returns a double
  double PGnx = 0.0; // probability of growing at node nx
  std::vector<size_t> goodvars; // variables nx can split on
  size_t vi = 0; // index of split variable
  size_t v = 0; // actual variable we're splitting on
  int L,U;
  size_t c = 0; // cut-point
  size_t max_depth = 0;
  size_t prev_max_depth = 0; // from previous round what is the max depth
  bool flag = true; // flag that indicates that we can continue trying to grow tree
  int counter = 0; // a cluge to make sure we don't grow the tree infinitely
  double unif = 0.0;
  while(flag && counter < 100){
    // find the valid bottom nodes
    Rcout << "counter = " << counter << endl;
    goodbots.clear();
    prev_max_depth = max_depth;
    
    // next several lines are copied directly from the function getpb
    bnv.clear(); // all the bottom nodes
    x.getbots(bnv);
    for(size_t i = 0; i != bnv.size(); i++){
      if(cansplit(bnv[i],xi)) goodbots.push_back(bnv[i]);
    }
 
    Rcout << "goodbots.size() = " << goodbots.size() << endl;
    if(goodbots.size() > 0){
      for(size_t ni = 0; ni != goodbots.size();ni++){
        dnx = goodbots[ni]->depth(); // depth of (ni)^th bottom node
        if(dnx > max_depth) max_depth = dnx;
      }
      Rcout << "max_depth = " << max_depth << endl;
      if(max_depth < prev_max_depth){
        Rcout << "max_depth is less than prev_max_depth. Something went horribly wrong."<< endl;
        break;
      } else if(max_depth > 1 + prev_max_depth){
        Rcout << "max_depth is greater than 1+prev_max_depth. Something went wrong." << endl;
        break;
      } else if( (max_depth == prev_max_depth) && (max_depth != 0)){
        Rcout << "max_depth = prev_max_depth = " << prev_max_depth;
        Rcout << " ... Can't try to grow further." << endl;
        break;
      } else if( (max_depth == prev_max_depth + 1) || (max_depth == 0)){
        flag = false; // re-set the value of the flag.
        for(size_t ni = 0; ni != goodbots.size();ni++){
          nx = goodbots[ni];
          dnx = nx->depth();
          if(dnx == max_depth){ // only grow at nodes which are at maximum depth
            goodvars.clear();
            getgoodvars(nx, xi, goodvars); // get the variables we're allowed to split on
            if(goodvars.size() > 0){
              Rcout << "good var size = " << goodvars.size() ;
              vi = floor(gen.uniform()*goodvars.size());
              v = goodvars[vi];
              Rcout << "  v = " << v ;
              L = 0;
              U = xi[v].size()-1;
              nx->rg(v, &L,&U);
              c = L + floor(gen.uniform()*(U-L+1)); // U-L+1 is number of available split points
              Rcout << "  c = " << c << endl;
              PGnx = alpha/pow(1.0 + dnx, beta);
              Rcout << " dnx = " << dnx << " PGnx = " << PGnx << " unif = ";
              unif = gen.uniform();
              if(unif < PGnx){
                Rcout << unif << " ... was able to grow this node!" << endl;
                flag = true; // we can continue growing the tree in the next iteration
                // do birth: set mul and mur to zero for now.
                x.birth(nx->nid(),v,c,0.0,0.0);
              } else{
                Rcout << unif << " ... unable to grow this node!" << endl;
              }
            } else{
              Rcout << "No good variables to split on" << endl;
              break;
            }
          } // closes if checking whether bottom node nx is at maximum depth
        } // closes loop over the nodes
      } // closes if checking whether max_depth = 1 + prev_max_depth (or that max_depth = 0)
    }// closes if checking if goodbots is empty
    counter++;
  } // closes while
  
  
  
  
  

}
