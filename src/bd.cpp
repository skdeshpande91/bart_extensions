#include<RcppArmadillo.h>

#include <iostream>

#include "info.h"
#include "tree.h"
#include "bd.h"
#include "funs.h"


using std::cout;
using std::endl;
using namespace arma;
using namespace Rcpp;
/*
notation: (as in old code): going from state x to state y (eg, incoming tree is x).

note: rather than have x and making a tree y
we just figure out what we need from x, the drawn bottom node,the drawn (v,c).
note sure what the right thing to do is.
Could make y (using a birth) and figure stuff out from y.
That is how the old code works.
*/

//bool bd(tree& x, arma::mat const &Omega, xinfo& xi, dinfo& di, pinfo& pi, RNG& gen)
/*
double bd_multi(tree &x, const arma::mat &Omega, xinfo &xi, dinfo &di, pinfo &pi, RNG &gen)
{
  tree::npv goodbots;                    //nodes we could birth at (split on)
  double PBx = getpb(x,xi,pi,goodbots);  //prob of a birth at x
                                         //Rcpp::Rcout << "[bd]: PBx = " << PBx << "  size(goodbots) = " << goodbots.size() <<endl;
  double unif = 0.0;
  
  // If statement for selecting birth or death proposal.
  unif = gen.uniform();
  if(unif < PBx) {
    //--------------------------------------------------
    // BIRTH PROPOSAL
    //--------------------------------------------------
    
    //Rcpp::Rcout << "[bd]: unif = " << unif << " ... propose birth" << endl;
    
    //--------------------------------------------------
    //draw proposal
    
    //draw bottom node, choose node index ni from list in goodbots
    size_t ni = floor(gen.uniform()*goodbots.size());
    //Rcpp::Rcout << "[bd]: proposing birth at bottom node " << ni << endl;
    tree::tree_p nx = goodbots[ni]; //the bottom node we might birth at
    
    //draw v,  the variable
    std::vector<size_t> goodvars; //variables nx can split on
    getgoodvars(nx,xi,goodvars);
    size_t vi = floor(gen.uniform()*goodvars.size()); //index of chosen split variable
    size_t v = goodvars[vi];
    //draw c, the cutpoint
    int L,U;
    L=0; U = xi[v].size()-1;
    nx->rg(v,&L,&U);
    size_t c = L + floor(gen.uniform()*(U-L+1)); //U-L+1 is number of available split points
                                                 //Rcpp::Rcout << "[bd]: proposed (v,c) = (" << v << "," << c << ")";
    
    //--------------------------------------------------
    //compute things needed for metropolis ratio
    double Pbotx = 1.0/goodbots.size(); //proposal prob of choosing nx
    size_t dnx = nx->depth();
    double PGnx = pi.alpha/pow(1.0 + dnx,pi.beta); //prior prob of growing at nx
                                                   //Rcpp::Rcout << "PGnx = " << PGnx << endl;
    
    double PGly, PGry; //prior probs of growing at new children (l and r) of proposal
    if(goodvars.size()>1) { //know there are variables we could split l and r on
      PGly = pi.alpha/pow(1.0 + dnx+1.0,pi.beta); //depth of new nodes would be one more
      PGry = PGly;
    } else { //only had v to work with, if it is exhausted at either child need PG=0
      if((int)(c-1)<L) { //v exhausted in new left child l, new upper limit would be c-1
        PGly = 0.0;
      } else {
        PGly = pi.alpha/pow(1.0 + dnx+1.0,pi.beta);
      }
      if(U < (int)(c+1)) { //v exhausted in new right child r, new lower limit would be c+1
        PGry = 0.0;
      } else {
        PGry = pi.alpha/pow(1.0 + dnx+1.0,pi.beta);
      }
    }
    
    //Rcpp::Rcout << "[bd]:   PGly = " << PGly << " PGry = " << PGry << endl;
    
    double PDy; //prob of proposing death at y
    if(goodbots.size()>1) { //can birth at y because splittable nodes left
      PDy = 1.0 - pi.pb;
    } else { //nx was the only node you could split on
      if((PGry==0) && (PGly==0)) { //cannot birth at y
        PDy=1.0;
      } else { //y can birth at either l or r
        PDy = 1.0 - pi.pb;
      }
    }
    
    //Rcpp::Rcout << "[bd]:   PDy = " << PDy << endl;
    
    double Pnogy; //death prob of choosing the nog node at y
    size_t nnogs = x.nnogs();
    tree::tree_cp nxp = nx->getp();
    if(nxp==0) { //no parent, nx is the top and only node
      Pnogy=1.0;
    } else {
      //if(nxp->ntype() == 'n') { //if parent is a nog, number of nogs same at x and y
      if(nxp->isnog()) { //if parent is a nog, number of nogs same at x and y
        Pnogy = 1.0/nnogs;
      } else { //if parent is not a nog, y has one more nog.
        Pnogy = 1.0/(nnogs+1.0);
      }
    }
    //Rcpp::Rcout << "[bd]: Pnogy = " << Pnogy << endl;
    
    //--------------------------------------------------
    //compute sufficient statistics
    sinfo sl;
    sinfo sr;
    sinfo st;
    getsuff(x, nx, v, c, xi, di, sl, sr);
    st.n = sl.n + sr.n;
    // concatenate I from sl and sr
    st.I.clear();
    st.I.reserve(sl.I.size() + sr.I.size());
    st.I.insert(st.I.end(), sl.I.begin(), sl.I.end());
    st.I.insert(st.I.end(), sr.I.begin(), sr.I.end());
    
    //--------------------------------------------------
    //compute alpha
    double alpha=0.0,alpha1=0.0,alpha2=0.0;
    double lill=0.0,lilr=0.0,lilt=0.0;
    double mu_bar_l = 0.0, V_l = 0.0; // posterior mean, variance of mu for left child
    double mu_bar_r = 0.0, V_r = 0.0; // posterior mean, variance of mu for right child
    double mu_bar_t = 0.0, V_t = 0.0; // posterior mean, variance of mu for combined node
                                      // need to define sigma0!!
    
    if((sl.n>=5) && (sr.n>=5)) { //kludge? [SKD]: why do we have this requirement??
      //get_mu_post_param(mu_bar_l, V_l, Omega, sl, di, pi.sigma_mu[di.k]);
      //get_mu_post_param(mu_bar_r, V_r, Omega, sr, di, pi.sigma_mu[di.k]);
      //get_mu_post_param(mu_bar_t, V_t, Omega, st, di, pi.sigma_mu[di.k]);
      mu_posterior_multi(mu_bar_l, V_l, Omega, sl, di, pi.sigma_mu[di.k]);
      mu_posterior_multi(mu_bar_r, V_r, Omega, sr, di, pi.sigma_mu[di.k]);
      mu_posterior_multi(mu_bar_t, V_t, Omega, st, di, pi.sigma_mu[di.k]);
      
      lill = 0.5 * log(V_l) + 0.5 * (mu_bar_l * mu_bar_l)/V_l;
      lilr = 0.5 * log(V_r) + 0.5 * (mu_bar_r * mu_bar_r)/V_r;
      lilt = 0.5 * log(V_t) + 0.5 * (mu_bar_t * mu_bar_t)/V_t;
      //Rcpp::Rcout << "[bd]:    pi.sigma0[" << di.k << "] = " << pi.sigma0[di.k] << endl;
      //Rcpp::Rcout << "[bd]:    st.I.size() = " << st.I.size() << endl;
      //Rcpp::Rcout << "[bd]:    mu_bar_l = " << mu_bar_l << "   V_l = " << V_l << "    lill = " << lill << endl;
      //Rcpp::Rcout << "[bd]:    mu_bar_r = " << mu_bar_r << "   V_r = " << V_r << "    lilr = " << lilr << endl;
      //Rcpp::Rcout << "[bd]:    mu_bar_t = " << mu_bar_t << "   V_t = " << V_t << "    lilt = " << lilt << endl;
      
      alpha1 = (PGnx*(1.0-PGly)*(1.0-PGry)*PDy*Pnogy)/((1.0-PGnx)*PBx*Pbotx);
      alpha2 = alpha1*exp(lill+lilr-lilt) * 1.0/pi.sigma_mu[di.k];
      alpha = std::min(1.0,alpha2);
    } else {
      //Rcpp::Rcout << "[bd]:   One of left and right child has fewer than 5 observations. Setting alpha = 0" << endl;
      alpha=0.0;
    }
    //Rcpp::Rcout << "  alpha = " << alpha;
    //alpha_track[k][t].push_back(alpha);
    //--------------------------------------------------
    //finally ready to try metrop
    //--------------------------------------------------
    
    unif = gen.uniform();
    //Rcpp::Rcout << "    unif = " << unif << endl;
    if(unif < alpha) {
      
      //--------------------------------------------------
      // do birth:
      // For the moment, we will set the new terminal node values to zero
      // Since we immediately fill them in the next step of the MCMC with drmu.
      // Set mul and mur to zero, since we will immediately
      
      x.birth(nx->nid(),v,c,0.0,0.0);
      
      return(alpha);
    } else {
      return(alpha);
    }
  } else {
    //--------------------------------------------------
    // DEATH PROPOSAL
    //--------------------------------------------------
    
    //Rcpp::Rcout << "[bd]: propose death" << endl;
    
    //--------------------------------------------------
    //draw proposal
    
    //draw nog node, any nog node is a possibility
    tree::npv nognds; //nog nodes
    x.getnogs(nognds);
    size_t ni = floor(gen.uniform()*nognds.size());
    //Rcpp::Rcout << "[bd]: proposing death at bottom node " << ni << endl;
    tree::tree_p nx = nognds[ni]; //the nog node we might kill children at
                                  //Rcpp::Rcout << "bd: " << "draw proposal" << endl;
    
    //--------------------------------------------------
    //compute things needed for metropolis ratio
    
    double PGny; //prob the nog node grows
    size_t dny = nx->depth();
    PGny = pi.alpha/pow(1.0+dny,pi.beta);
    //Rcpp::Rcout << "[bd]:    PGny = " << PGny << endl;
    
    //better way to code these two?
    double PGlx = pgrow(nx->getl(),xi,pi);
    double PGrx = pgrow(nx->getr(),xi,pi);
    
    double PBy;  //prob of birth move at y
                 //if(nx->ntype()=='t') { //is the nog node nx the top node
    if(!(nx->p)) { //is the nog node nx the top node
      PBy = 1.0;
    } else {
      PBy = pi.pb;
    }
    
    double Pboty;  //prob of choosing the nog as bot to split on when y
    int ngood = goodbots.size();
    if(cansplit(nx->getl(),xi)) --ngood; //if can split at left child, lose this one
    if(cansplit(nx->getr(),xi)) --ngood; //if can split at right child, lose this one
    ++ngood;  //know you can split at nx
    Pboty=1.0/ngood;
    
    double PDx = 1.0-PBx; //prob of a death step at x
    double Pnogx = 1.0/nognds.size();
    
    //Rcpp::Rcout << "bd: " << "compute things needed for MH ratio" << endl;
    
    //--------------------------------------------------
    //compute sufficient statistics
    sinfo sl;
    sinfo sr;
    sinfo st;
    getsuff(x, nx->getl(),nx->getr(),xi,di,sl,sr);
    st.n = sl.n + sr.n;
    // Concatentate sl.I and sr.I into st.I
    st.I.clear();
    st.I.reserve(sl.I.size() + sr.I.size());
    st.I.insert(st.I.end(), sl.I.begin(), sl.I.end());
    st.I.insert(st.I.end(), sr.I.begin(), sr.I.end());
    
    double mu_bar_l = 0.0, mu_bar_r = 0.0, mu_bar_t = 0.0;
    double V_l = 0.0, V_r = 0.0, V_t = 0.0;
    
    mu_posterior_multi(mu_bar_l, V_l, Omega, sl, di, pi.sigma_mu[di.k]);
    mu_posterior_multi(mu_bar_r, V_r, Omega, sr, di, pi.sigma_mu[di.k]);
    mu_posterior_multi(mu_bar_t, V_t, Omega, st, di, pi.sigma_mu[di.k]);
    
    double lill = 0.5 * log(V_l) + 0.5*(mu_bar_l * mu_bar_l)/V_l;
    double lilr = 0.5 * log(V_r) + 0.5*(mu_bar_r * mu_bar_r)/V_r;
    double lilt = 0.5 * log(V_t) + 0.5*(mu_bar_t * mu_bar_t)/V_t;
    //Rcpp::Rcout << "[bd]:    pi.sigma0[" << di.k << "] = " << pi.sigma0[di.k] << endl;
    //Rcpp::Rcout << "[bd]:    st.I.size() = " << st.I.size() << endl;
    //Rcpp::Rcout << "[bd]:    mu_bar_l = " << mu_bar_l << "   V_l = " << V_l << "    lill = " << lill << endl;
    //Rcpp::Rcout << "[bd]:    mu_bar_r = " << mu_bar_r << "   V_r = " << V_r << "    lilr = " << lilr << endl;
    //Rcpp::Rcout << "[bd]:    mu_bar_t = " << mu_bar_t << "   V_t = " << V_t << "    lilt = " << lilt << endl;
    
    double alpha1 = ((1.0-PGny)*PBy*Pboty)/(PGny*(1.0-PGlx)*(1.0-PGrx)*PDx*Pnogx);
    double alpha2 = alpha1*exp(lilt - lill - lilr) * pi.sigma_mu[di.k];
    double alpha = std::min(1.0,alpha2);
    //Rcpp::Rcout << "[bd]:    alpha = " << alpha ;
    //alpha_track[k][t].push_back(alpha);
    //--------------------------------------------------
    //finally ready to try metrop
    unif = gen.uniform();
    //Rcpp::Rcout << "    unif = " << unif << endl;
    if(unif<alpha) {
      // Do the death move. We will set the new mu parameter to be 0.0
      // This will immediately be overwritten in the next step when we call drmu
      //Rcpp::Rcout << " ... death successful" << endl;
      x.death(nx->nid(),0.0);
      //return true;
      return(-1.0*alpha);
    } else {
      return(-1.0*alpha);
    }
  } // closes else for the death proposal
}
*/

double bd_uni(tree &x, const double &omega, xinfo &xi, dinfo &di, pinfo &pi, RNG &gen)
{
  tree::npv goodbots;                    //nodes we could birth at (split on)
  double PBx = getpb(x,xi,pi,goodbots);  //prob of a birth at x
                                         //Rcpp::Rcout << "[bd]: PBx = " << PBx << "  size(goodbots) = " << goodbots.size() <<endl;
  double unif = 0.0;
  
  // If statement for selecting birth or death proposal.
  unif = gen.uniform();
  if(unif < PBx) {
    //--------------------------------------------------
    // BIRTH PROPOSAL
    //--------------------------------------------------
    
    //Rcpp::Rcout << "[bd]: unif = " << unif << " ... propose birth" << endl;
    
    //--------------------------------------------------
    //draw proposal
    
    //draw bottom node, choose node index ni from list in goodbots
    size_t ni = floor(gen.uniform()*goodbots.size());
    //Rcpp::Rcout << "[bd]: proposing birth at bottom node " << ni << endl;
    tree::tree_p nx = goodbots[ni]; //the bottom node we might birth at
    
    //draw v,  the variable
    std::vector<size_t> goodvars; //variables nx can split on
    getgoodvars(nx,xi,goodvars);
    size_t vi = floor(gen.uniform()*goodvars.size()); //index of chosen split variable
    size_t v = goodvars[vi];
    //draw c, the cutpoint
    int L,U;
    L=0; U = xi[v].size()-1;
    nx->rg(v,&L,&U);
    size_t c = L + floor(gen.uniform()*(U-L+1)); //U-L+1 is number of available split points
                                                 //Rcpp::Rcout << "[bd]: proposed (v,c) = (" << v << "," << c << ")";
    
    //--------------------------------------------------
    //compute things needed for metropolis ratio
    double Pbotx = 1.0/goodbots.size(); //proposal prob of choosing nx
    size_t dnx = nx->depth();
    double PGnx = pi.alpha/pow(1.0 + dnx,pi.beta); //prior prob of growing at nx
                                                   //Rcpp::Rcout << "PGnx = " << PGnx << endl;
    
    double PGly, PGry; //prior probs of growing at new children (l and r) of proposal
    if(goodvars.size()>1) { //know there are variables we could split l and r on
      PGly = pi.alpha/pow(1.0 + dnx+1.0,pi.beta); //depth of new nodes would be one more
      PGry = PGly;
    } else { //only had v to work with, if it is exhausted at either child need PG=0
      if((int)(c-1)<L) { //v exhausted in new left child l, new upper limit would be c-1
        PGly = 0.0;
      } else {
        PGly = pi.alpha/pow(1.0 + dnx+1.0,pi.beta);
      }
      if(U < (int)(c+1)) { //v exhausted in new right child r, new lower limit would be c+1
        PGry = 0.0;
      } else {
        PGry = pi.alpha/pow(1.0 + dnx+1.0,pi.beta);
      }
    }
    
    //Rcpp::Rcout << "[bd]:   PGly = " << PGly << " PGry = " << PGry << endl;
    
    double PDy; //prob of proposing death at y
    if(goodbots.size()>1) { //can birth at y because splittable nodes left
      PDy = 1.0 - pi.pb;
    } else { //nx was the only node you could split on
      if((PGry==0) && (PGly==0)) { //cannot birth at y
        PDy=1.0;
      } else { //y can birth at either l or r
        PDy = 1.0 - pi.pb;
      }
    }
    
    //Rcpp::Rcout << "[bd]:   PDy = " << PDy << endl;
    
    double Pnogy; //death prob of choosing the nog node at y
    size_t nnogs = x.nnogs();
    tree::tree_cp nxp = nx->getp();
    if(nxp==0) { //no parent, nx is the top and only node
      Pnogy=1.0;
    } else {
      //if(nxp->ntype() == 'n') { //if parent is a nog, number of nogs same at x and y
      if(nxp->isnog()) { //if parent is a nog, number of nogs same at x and y
        Pnogy = 1.0/nnogs;
      } else { //if parent is not a nog, y has one more nog.
        Pnogy = 1.0/(nnogs+1.0);
      }
    }
    //Rcpp::Rcout << "[bd]: Pnogy = " << Pnogy << endl;
    
    //--------------------------------------------------
    //compute sufficient statistics
    sinfo sl;
    sinfo sr;
    sinfo st;
    getsuff(x, nx, v, c, xi, di, sl, sr);
    st.n = sl.n + sr.n;
    // concatenate I from sl and sr
    st.I.clear();
    st.I.reserve(sl.I.size() + sr.I.size());
    st.I.insert(st.I.end(), sl.I.begin(), sl.I.end());
    st.I.insert(st.I.end(), sr.I.begin(), sr.I.end());
    
    //--------------------------------------------------
    //compute alpha
    double alpha=0.0,alpha1=0.0,alpha2=0.0;
    double lill=0.0,lilr=0.0,lilt=0.0;
    double mu_bar_l = 0.0, V_l = 0.0; // posterior mean, variance of mu for left child
    double mu_bar_r = 0.0, V_r = 0.0; // posterior mean, variance of mu for right child
    double mu_bar_t = 0.0, V_t = 0.0; // posterior mean, variance of mu for combined node
                                      // need to define sigma0!!
    
    if((sl.n>=5) && (sr.n>=5)) { //kludge? [SKD]: why do we have this requirement??
                                 //get_mu_post_param(mu_bar_l, V_l, Omega, sl, di, pi.sigma_mu[di.k]);
                                 //get_mu_post_param(mu_bar_r, V_r, Omega, sr, di, pi.sigma_mu[di.k]);
                                 //get_mu_post_param(mu_bar_t, V_t, Omega, st, di, pi.sigma_mu[di.k]);
      mu_posterior_uni(mu_bar_l, V_l, omega, sl, di, pi.sigma_mu[di.k]);
      mu_posterior_uni(mu_bar_r, V_r, omega, sr, di, pi.sigma_mu[di.k]);
      mu_posterior_uni(mu_bar_t, V_t, omega, st, di, pi.sigma_mu[di.k]);
      
      lill = 0.5 * log(V_l) + 0.5 * (mu_bar_l * mu_bar_l)/V_l;
      lilr = 0.5 * log(V_r) + 0.5 * (mu_bar_r * mu_bar_r)/V_r;
      lilt = 0.5 * log(V_t) + 0.5 * (mu_bar_t * mu_bar_t)/V_t;
      //Rcpp::Rcout << "[bd]:    pi.sigma0[" << di.k << "] = " << pi.sigma0[di.k] << endl;
      //Rcpp::Rcout << "[bd]:    st.I.size() = " << st.I.size() << endl;
      //Rcpp::Rcout << "[bd]:    mu_bar_l = " << mu_bar_l << "   V_l = " << V_l << "    lill = " << lill << endl;
      //Rcpp::Rcout << "[bd]:    mu_bar_r = " << mu_bar_r << "   V_r = " << V_r << "    lilr = " << lilr << endl;
      //Rcpp::Rcout << "[bd]:    mu_bar_t = " << mu_bar_t << "   V_t = " << V_t << "    lilt = " << lilt << endl;
      
      alpha1 = (PGnx*(1.0-PGly)*(1.0-PGry)*PDy*Pnogy)/((1.0-PGnx)*PBx*Pbotx);
      alpha2 = alpha1*exp(lill+lilr-lilt) * 1.0/pi.sigma_mu[di.k];
      alpha = std::min(1.0,alpha2);
    } else {
      //Rcpp::Rcout << "[bd]:   One of left and right child has fewer than 5 observations. Setting alpha = 0" << endl;
      alpha=0.0;
    }
    //Rcpp::Rcout << "  alpha = " << alpha;
    //alpha_track[k][t].push_back(alpha);
    //--------------------------------------------------
    //finally ready to try metrop
    //--------------------------------------------------
    
    unif = gen.uniform();
    //Rcpp::Rcout << "    unif = " << unif << endl;
    if(unif < alpha) {
      
      //--------------------------------------------------
      // do birth:
      // For the moment, we will set the new terminal node values to zero
      // Since we immediately fill them in the next step of the MCMC with drmu.
      // Set mul and mur to zero, since we will immediately
      
      x.birth(nx->nid(),v,c,0.0,0.0);
      
      return(alpha);
    } else {
      return(alpha);
    }
  } else {
    //--------------------------------------------------
    // DEATH PROPOSAL
    //--------------------------------------------------
    
    //Rcpp::Rcout << "[bd]: propose death" << endl;
    
    //--------------------------------------------------
    //draw proposal
    
    //draw nog node, any nog node is a possibility
    tree::npv nognds; //nog nodes
    x.getnogs(nognds);
    size_t ni = floor(gen.uniform()*nognds.size());
    //Rcpp::Rcout << "[bd]: proposing death at bottom node " << ni << endl;
    tree::tree_p nx = nognds[ni]; //the nog node we might kill children at
                                  //Rcpp::Rcout << "bd: " << "draw proposal" << endl;
    
    //--------------------------------------------------
    //compute things needed for metropolis ratio
    
    double PGny; //prob the nog node grows
    size_t dny = nx->depth();
    PGny = pi.alpha/pow(1.0+dny,pi.beta);
    //Rcpp::Rcout << "[bd]:    PGny = " << PGny << endl;
    
    //better way to code these two?
    double PGlx = pgrow(nx->getl(),xi,pi);
    double PGrx = pgrow(nx->getr(),xi,pi);
    
    double PBy;  //prob of birth move at y
                 //if(nx->ntype()=='t') { //is the nog node nx the top node
    if(!(nx->p)) { //is the nog node nx the top node
      PBy = 1.0;
    } else {
      PBy = pi.pb;
    }
    
    double Pboty;  //prob of choosing the nog as bot to split on when y
    int ngood = goodbots.size();
    if(cansplit(nx->getl(),xi)) --ngood; //if can split at left child, lose this one
    if(cansplit(nx->getr(),xi)) --ngood; //if can split at right child, lose this one
    ++ngood;  //know you can split at nx
    Pboty=1.0/ngood;
    
    double PDx = 1.0-PBx; //prob of a death step at x
    double Pnogx = 1.0/nognds.size();
    
    //Rcpp::Rcout << "bd: " << "compute things needed for MH ratio" << endl;
    
    //--------------------------------------------------
    //compute sufficient statistics
    sinfo sl;
    sinfo sr;
    sinfo st;
    getsuff(x, nx->getl(),nx->getr(),xi,di,sl,sr);
    st.n = sl.n + sr.n;
    // Concatentate sl.I and sr.I into st.I
    st.I.clear();
    st.I.reserve(sl.I.size() + sr.I.size());
    st.I.insert(st.I.end(), sl.I.begin(), sl.I.end());
    st.I.insert(st.I.end(), sr.I.begin(), sr.I.end());
    
    double mu_bar_l = 0.0, mu_bar_r = 0.0, mu_bar_t = 0.0;
    double V_l = 0.0, V_r = 0.0, V_t = 0.0;
    
    mu_posterior_uni(mu_bar_l, V_l, omega, sl, di, pi.sigma_mu[di.k]);
    mu_posterior_uni(mu_bar_r, V_r, omega, sr, di, pi.sigma_mu[di.k]);
    mu_posterior_uni(mu_bar_t, V_t, omega, st, di, pi.sigma_mu[di.k]);
    
    double lill = 0.5 * log(V_l) + 0.5*(mu_bar_l * mu_bar_l)/V_l;
    double lilr = 0.5 * log(V_r) + 0.5*(mu_bar_r * mu_bar_r)/V_r;
    double lilt = 0.5 * log(V_t) + 0.5*(mu_bar_t * mu_bar_t)/V_t;
    //Rcpp::Rcout << "[bd]:    pi.sigma0[" << di.k << "] = " << pi.sigma0[di.k] << endl;
    //Rcpp::Rcout << "[bd]:    st.I.size() = " << st.I.size() << endl;
    //Rcpp::Rcout << "[bd]:    mu_bar_l = " << mu_bar_l << "   V_l = " << V_l << "    lill = " << lill << endl;
    //Rcpp::Rcout << "[bd]:    mu_bar_r = " << mu_bar_r << "   V_r = " << V_r << "    lilr = " << lilr << endl;
    //Rcpp::Rcout << "[bd]:    mu_bar_t = " << mu_bar_t << "   V_t = " << V_t << "    lilt = " << lilt << endl;
    
    double alpha1 = ((1.0-PGny)*PBy*Pboty)/(PGny*(1.0-PGlx)*(1.0-PGrx)*PDx*Pnogx);
    double alpha2 = alpha1*exp(lilt - lill - lilr) * pi.sigma_mu[di.k];
    double alpha = std::min(1.0,alpha2);
    //Rcpp::Rcout << "[bd]:    alpha = " << alpha ;
    //alpha_track[k][t].push_back(alpha);
    //--------------------------------------------------
    //finally ready to try metrop
    unif = gen.uniform();
    //Rcpp::Rcout << "    unif = " << unif << endl;
    if(unif<alpha) {
      // Do the death move. We will set the new mu parameter to be 0.0
      // This will immediately be overwritten in the next step when we call drmu
      //Rcpp::Rcout << " ... death successful" << endl;
      x.death(nx->nid(),0.0);
      //return true;
      return(-1.0*alpha);
    } else {
      return(-1.0*alpha);
    }
  } // closes else for the death proposal
}

// overloaded to work with the new prior info classes
double bd_uni(tree &x, const double &sigma, xinfo &xi, data_info &di, tree_prior_info &tree_pi, RNG &gen)
{
  tree::npv goodbots; //nodes we could birth at (split on)
  double PBx = getpb(x,xi,tree_pi,goodbots);  //prob of a birth at x
  double unif = 0.0;
  
  // If statement for selecting birth or death proposal.
  unif = gen.uniform();
  if(unif < PBx) {
    //--------------------------------------------------
    // BIRTH PROPOSAL
    //--------------------------------------------------

    //draw proposal
    
    //draw bottom node, choose node index ni from list in goodbots
    size_t ni = floor(gen.uniform()*goodbots.size());
    //Rcpp::Rcout << "[bd]: proposing birth at bottom node " << ni << endl;
    tree::tree_p nx = goodbots[ni]; //the bottom node we might birth at
    
    //draw v,  the variable we split on
    std::vector<size_t> goodvars; //variables nx can split on
    getgoodvars(nx,xi,goodvars);
    size_t vi = floor(gen.uniform()*goodvars.size()); //index of chosen split variable
    size_t v = goodvars[vi];
    //draw c, the cutpoint
    int L,U;
    L=0; U = xi[v].size()-1;
    nx->rg(v,&L,&U);
    size_t c = L + floor(gen.uniform()*(U-L+1)); //U-L+1 is number of available split points
    //Rcpp::Rcout << "[bd]: proposed (v,c) = (" << v << "," << c << ")";
    
    //--------------------------------------------------
    //compute things needed for metropolis ratio
    double Pbotx = 1.0/goodbots.size(); //proposal prob of choosing nx
    size_t dnx = nx->depth();
    double PGnx = tree_pi.alpha/pow(1.0 + dnx,tree_pi.beta); //prior prob of growing at nx
    //Rcpp::Rcout << "PGnx = " << PGnx << endl;
    
    double PGly, PGry; //prior probs of growing at new children (l and r) of proposal
    if(goodvars.size()>1) { //know there are variables we could split l and r on
      PGly = tree_pi.alpha/pow(1.0 + dnx+1.0,tree_pi.beta); //depth of new nodes would be one more
      PGry = PGly;
    } else { //only had v to work with, if it is exhausted at either child need PG=0
      if((int)(c-1)<L) { //v exhausted in new left child l, new upper limit would be c-1
        PGly = 0.0;
      } else {
        PGly = tree_pi.alpha/pow(1.0 + dnx+1.0,tree_pi.beta);
      }
      if(U < (int)(c+1)) { //v exhausted in new right child r, new lower limit would be c+1
        PGry = 0.0;
      } else {
        PGry = tree_pi.alpha/pow(1.0 + dnx+1.0,tree_pi.beta);
      }
    }
    
    //Rcpp::Rcout << "[bd]:   PGly = " << PGly << " PGry = " << PGry << endl;
    
    double PDy; //prob of proposing death at y
    if(goodbots.size()>1) { //can birth at y because splittable nodes left
      PDy = 1.0 - tree_pi.pb;
    } else { //nx was the only node you could split on
      if((PGry==0) && (PGly==0)) { //cannot birth at y
        PDy=1.0;
      } else { //y can birth at either l or r
        PDy = 1.0 - tree_pi.pb;
      }
    }
    
    //Rcpp::Rcout << "[bd]:   PDy = " << PDy << endl;
    
    double Pnogy; //death prob of choosing the nog node at y
    size_t nnogs = x.nnogs();
    tree::tree_cp nxp = nx->getp();
    if(nxp==0) { //no parent, nx is the top and only node
      Pnogy=1.0;
    } else {
      //if(nxp->ntype() == 'n') { //if parent is a nog, number of nogs same at x and y
      if(nxp->isnog()) { //if parent is a nog, number of nogs same at x and y
        Pnogy = 1.0/nnogs;
      } else { //if parent is not a nog, y has one more nog.
        Pnogy = 1.0/(nnogs+1.0);
      }
    }
    //Rcpp::Rcout << "[bd]: Pnogy = " << Pnogy << endl;
    
    //--------------------------------------------------
    //compute sufficient statistics
    sinfo sl;
    sinfo sr;
    sinfo st;
    getsuff(x, nx, v, c, xi, di, sl, sr);
    st.n = sl.n + sr.n;
    // concatenate I from sl and sr
    st.I.clear();
    st.I.reserve(sl.I.size() + sr.I.size());
    st.I.insert(st.I.end(), sl.I.begin(), sl.I.end());
    st.I.insert(st.I.end(), sr.I.begin(), sr.I.end());
    
    //--------------------------------------------------
    //compute alpha
    double alpha=0.0,alpha1=0.0,alpha2=0.0;
    double lill=0.0,lilr=0.0,lilt=0.0;
    double mu_bar_l = 0.0, V_l = 0.0; // posterior mean, variance of mu for left child
    double mu_bar_r = 0.0, V_r = 0.0; // posterior mean, variance of mu for right child
    double mu_bar_t = 0.0, V_t = 0.0; // posterior mean, variance of mu for combined node
    
    if((sl.n>=5) && (sr.n>=5)) { //kluge? [SKD]: why do we have this requirement??
      mu_posterior_uni(mu_bar_l, V_l, sigma, sl, di, tree_pi);
      mu_posterior_uni(mu_bar_r, V_r, sigma, sr, di, tree_pi);
      mu_posterior_uni(mu_bar_t, V_t, sigma, st, di, tree_pi);
      
      lill = 0.5 * log(V_l) + 0.5 * (mu_bar_l * mu_bar_l)/V_l;
      lilr = 0.5 * log(V_r) + 0.5 * (mu_bar_r * mu_bar_r)/V_r;
      lilt = 0.5 * log(V_t) + 0.5 * (mu_bar_t * mu_bar_t)/V_t;

      alpha1 = (PGnx*(1.0-PGly)*(1.0-PGry)*PDy*Pnogy)/((1.0-PGnx)*PBx*Pbotx);
      alpha2 = alpha1*exp(lill+lilr-lilt) * 1.0/tree_pi.sigma_mu;
      alpha = std::min(1.0,alpha2);
    } else {
      alpha=0.0;
    }
    //Rcpp::Rcout << "  alpha = " << alpha;
    //alpha_track[k][t].push_back(alpha);
    //--------------------------------------------------
    //finally ready to try metrop
    //--------------------------------------------------
    
    unif = gen.uniform();
    //Rcpp::Rcout << "    unif = " << unif << endl;
    if(unif < alpha) {
      
      //--------------------------------------------------
      // do birth:
      // For the moment, we will set the new terminal node values to zero
      // Since we immediately fill them in the next step of the MCMC with drmu.
      // Set mul and mur to zero, since we will immediately
      
      x.birth(nx->nid(),v,c,0.0,0.0);
      
      return(alpha);
    } else {
      return(alpha);
    }
  } else {
    //--------------------------------------------------
    // DEATH PROPOSAL
    //--------------------------------------------------
    
    //Rcpp::Rcout << "[bd]: propose death" << endl;
    
    //--------------------------------------------------
    //draw proposal
    
    //draw nog node, any nog node is a possibility
    tree::npv nognds; //nog nodes
    x.getnogs(nognds);
    size_t ni = floor(gen.uniform()*nognds.size());
    //Rcpp::Rcout << "[bd]: proposing death at bottom node " << ni << endl;
    tree::tree_p nx = nognds[ni]; //the nog node we might kill children at
    //Rcpp::Rcout << "bd: " << "draw proposal" << endl;
    
    //--------------------------------------------------
    //compute things needed for metropolis ratio
    
    double PGny; //prob the nog node grows
    size_t dny = nx->depth();
    PGny = tree_pi.alpha/pow(1.0+dny,tree_pi.beta);
    //Rcpp::Rcout << "[bd]:    PGny = " << PGny << endl;
    
    //better way to code these two?
    double PGlx = pgrow(nx->getl(),xi,tree_pi);
    double PGrx = pgrow(nx->getr(),xi,tree_pi);
    
    double PBy;  //prob of birth move at y
    //if(nx->ntype()=='t') { //is the nog node nx the top node
    if(!(nx->p)) { //is the nog node nx the top node
      PBy = 1.0;
    } else {
      PBy = tree_pi.pb;
    }
    
    double Pboty;  //prob of choosing the nog as bot to split on when y
    int ngood = goodbots.size();
    if(cansplit(nx->getl(),xi)) --ngood; //if can split at left child, lose this one
    if(cansplit(nx->getr(),xi)) --ngood; //if can split at right child, lose this one
    ++ngood;  //know you can split at nx
    Pboty=1.0/ngood;
    
    double PDx = 1.0-PBx; //prob of a death step at x
    double Pnogx = 1.0/nognds.size();
    
    //Rcpp::Rcout << "bd: " << "compute things needed for MH ratio" << endl;
    
    //--------------------------------------------------
    //compute sufficient statistics
    sinfo sl;
    sinfo sr;
    sinfo st;
    getsuff(x, nx->getl(),nx->getr(),xi,di,sl,sr);
    st.n = sl.n + sr.n;
    // Concatentate sl.I and sr.I into st.I
    st.I.clear();
    st.I.reserve(sl.I.size() + sr.I.size());
    st.I.insert(st.I.end(), sl.I.begin(), sl.I.end());
    st.I.insert(st.I.end(), sr.I.begin(), sr.I.end());
    
    double mu_bar_l = 0.0, mu_bar_r = 0.0, mu_bar_t = 0.0;
    double V_l = 0.0, V_r = 0.0, V_t = 0.0;
    
    mu_posterior_uni(mu_bar_l, V_l, sigma, sl, di, tree_pi);
    mu_posterior_uni(mu_bar_r, V_r, sigma, sr, di, tree_pi);
    mu_posterior_uni(mu_bar_t, V_t, sigma, st, di, tree_pi);
    
    double lill = 0.5 * log(V_l) + 0.5*(mu_bar_l * mu_bar_l)/V_l;
    double lilr = 0.5 * log(V_r) + 0.5*(mu_bar_r * mu_bar_r)/V_r;
    double lilt = 0.5 * log(V_t) + 0.5*(mu_bar_t * mu_bar_t)/V_t;

    
    double alpha1 = ((1.0-PGny)*PBy*Pboty)/(PGny*(1.0-PGlx)*(1.0-PGrx)*PDx*Pnogx);
    double alpha2 = alpha1*exp(lilt - lill - lilr) * tree_pi.sigma_mu;
    double alpha = std::min(1.0,alpha2);
    //Rcpp::Rcout << "[bd]:    alpha = " << alpha ;
    //alpha_track[k][t].push_back(alpha);
    //--------------------------------------------------
    //finally ready to try metrop
    unif = gen.uniform();
    //Rcpp::Rcout << "    unif = " << unif << endl;
    if(unif<alpha) {
      // Do the death move. We will set the new mu parameter to be 0.0
      // This will immediately be overwritten in the next step when we call drmu
      //Rcpp::Rcout << " ... death successful" << endl;
      x.death(nx->nid(),0.0);
      //return true;
      return(-1.0*alpha);
    } else {
      return(-1.0*alpha);
    }
  } // closes else for the death proposal
}








double bd_slfm(tree &x, const arma::mat &Phi, const arma::vec &sigma, xinfo &xi, dinfo_slfm &di, pinfo_slfm &pi, RNG &gen)
{
  tree::npv goodbots;  //nodes we could birth at (split on)
  double PBx = getpb(x,xi,pi,goodbots);  //prob of a birth at x
  double unif = 0.0;
  unif = gen.uniform();
  //Rcpp::Rcout << "[bd_slfm]: unif = " << unif << " PBx = " << PBx << endl;
  if(unif < PBx){
    //Rcpp::Rcout << "[bd_slfm]: birth proposal" << endl;
    //draw bottom node, choose node index ni from list in goodbots
    size_t ni = floor(gen.uniform()*goodbots.size());
    //Rcpp::Rcout << "[bd]: proposing birth at bottom node " << ni << endl;
    tree::tree_p nx = goodbots[ni]; //the bottom node we might birth at
    
    //draw v,  the variable
    std::vector<size_t> goodvars; //variables nx can split on
    getgoodvars(nx,xi,goodvars);
    size_t vi = floor(gen.uniform()*goodvars.size()); //index of chosen split variable
    size_t v = goodvars[vi];
    //draw c, the cutpoint
    int L,U;
    L=0; U = xi[v].size()-1;
    nx->rg(v,&L,&U);
    size_t c = L + floor(gen.uniform()*(U-L+1)); //U-L+1 is number of available split points
    
    //Rcpp::Rcout << "[bd]: proposed (v,c) = (" << v << "," << c << ")" << endl;
    //--------------------------------------------------
    //compute things needed for metropolis ratio
    double Pbotx = 1.0/goodbots.size(); //proposal prob of choosing nx
    size_t dnx = nx->depth();
    double PGnx = pi.alpha/pow(1.0 + dnx,pi.beta); //prior prob of growing at nx
                                                   //Rcpp::Rcout << "PGnx = " << PGnx << endl;
    
    double PGly, PGry; //prior probs of growing at new children (l and r) of proposal
    if(goodvars.size()>1) { //know there are variables we could split l and r on
      PGly = pi.alpha/pow(1.0 + dnx+1.0,pi.beta); //depth of new nodes would be one more
      PGry = PGly;
    } else { //only had v to work with, if it is exhausted at either child need PG=0
      if((int)(c-1)<L) { //v exhausted in new left child l, new upper limit would be c-1
        PGly = 0.0;
      } else {
        PGly = pi.alpha/pow(1.0 + dnx+1.0,pi.beta);
      }
      if(U < (int)(c+1)) { //v exhausted in new right child r, new lower limit would be c+1
        PGry = 0.0;
      } else {
        PGry = pi.alpha/pow(1.0 + dnx+1.0,pi.beta);
      }
    }
    
    //Rcpp::Rcout << "[bd]:   PGly = " << PGly << " PGry = " << PGry << endl;
    
    double PDy; //prob of proposing death at y
    if(goodbots.size()>1) { //can birth at y because splittable nodes left
      PDy = 1.0 - pi.pb;
    } else { //nx was the only node you could split on
      if((PGry==0) && (PGly==0)) { //cannot birth at y
        PDy=1.0;
      } else { //y can birth at either l or r
        PDy = 1.0 - pi.pb;
      }
    }
    
    //Rcpp::Rcout << "[bd]:   PDy = " << PDy << endl;
    
    double Pnogy; //death prob of choosing the nog node at y
    size_t nnogs = x.nnogs();
    tree::tree_cp nxp = nx->getp();
    if(nxp==0) { //no parent, nx is the top and only node
      Pnogy=1.0;
    } else {
      //if(nxp->ntype() == 'n') { //if parent is a nog, number of nogs same at x and y
      if(nxp->isnog()) { //if parent is a nog, number of nogs same at x and y
        Pnogy = 1.0/nnogs;
      } else { //if parent is not a nog, y has one more nog.
        Pnogy = 1.0/(nnogs+1.0);
      }
    }
    //Rcpp::Rcout << "[bd]: Ready to compute sufficient statistics" << endl;

    //--------------------------------------------------
    //compute sufficient statistics
    sinfo sl;
    sinfo sr;
    sinfo st;
    getsuff(x, nx, v, c, xi, di, sl, sr);
    //Rcpp::Rcout << "[bd]: Computed sufficient statistics" << endl;
    //Rcpp::Rcout << "left child: ";
    //for(int i = 0; i < sl.n; i++) Rcpp::Rcout << " " << sl.I[i] ;
    //Rcpp::Rcout << endl;
    
    //Rcpp::Rcout << "right child: ";
    //for(int i = 0; i < sr.n; i++) Rcpp::Rcout << " " << sr.I[i];
    //Rcpp::Rcout << endl;
    
    st.n = sl.n + sr.n;
    // concatenate I from sl and sr
    st.I.clear();
    st.I.reserve(sl.I.size() + sr.I.size());
    st.I.insert(st.I.end(), sl.I.begin(), sl.I.end());
    st.I.insert(st.I.end(), sr.I.begin(), sr.I.end());

    //--------------------------------------------------
    //compute alpha
    double alpha=0.0,alpha1=0.0,alpha2=0.0;
    double lill=0.0,lilr=0.0,lilt=0.0;
    double mu_bar_l = 0.0, V_l = 0.0; // posterior mean, variance of mu for left child
    double mu_bar_r = 0.0, V_r = 0.0; // posterior mean, variance of mu for right child
    double mu_bar_t = 0.0, V_t = 0.0; // posterior mean, variance of mu for combined node
    
    if((sl.n>=1) && (sr.n>=1)) { //kludge? [SKD]: why do we have this requirement??
      mu_posterior_slfm(mu_bar_l, V_l, Phi, sigma, sl, di, pi.sigma_mu[di.d]);
      mu_posterior_slfm(mu_bar_r, V_r, Phi, sigma, sr, di, pi.sigma_mu[di.d]);
      mu_posterior_slfm(mu_bar_t, V_t, Phi, sigma, st, di, pi.sigma_mu[di.d]);
      
      lill = 0.5 * log(V_l) + 0.5 * (mu_bar_l * mu_bar_l)/V_l;
      lilr = 0.5 * log(V_r) + 0.5 * (mu_bar_r * mu_bar_r)/V_r;
      lilt = 0.5 * log(V_t) + 0.5 * (mu_bar_t * mu_bar_t)/V_t;
 
      alpha1 = (PGnx*(1.0-PGly)*(1.0-PGry)*PDy*Pnogy)/((1.0-PGnx)*PBx*Pbotx);
      alpha2 = alpha1*exp(lill+lilr-lilt) * 1.0/pi.sigma_mu[di.d];
      alpha = std::min(1.0,alpha2);
    } else {
      //Rcpp::Rcout << "[bd]:   One of left and right child has fewer than 5 observations. Setting alpha = 0" << endl;
      alpha=0.0;
    }
    //Rcpp::Rcout << "  alpha = " << alpha;
    //--------------------------------------------------
    //finally ready to try metrop
    //--------------------------------------------------
    
    unif = gen.uniform();
    //Rcpp::Rcout << "    unif = " << unif << endl;
    if(unif < alpha) {
      
      //--------------------------------------------------
      // do birth:
      // For the moment, we will set the new terminal node values to zero
      // Since we immediately fill them in the next step of the MCMC with drmu.
      // Set mul and mur to zero, since we will immediately
      //Rcpp::Rcout << "    birth successful" << endl;
      x.birth(nx->nid(),v,c,0.0,0.0);
      
      return(alpha);
    } else {
      return(alpha);
    }
  } else{
    //--------------------------------------------------
    // DEATH PROPOSAL
    //--------------------------------------------------
    
    //Rcpp::Rcout << "[bd_slfm]: propose death" << endl;
    //--------------------------------------------------
    //draw proposal
    
    //draw nog node, any nog node is a possibility
    tree::npv nognds; //nog nodes
    x.getnogs(nognds);
    size_t ni = floor(gen.uniform()*nognds.size());
    //Rcpp::Rcout << "[bd]: proposing death at bottom node " << ni << endl;
    tree::tree_p nx = nognds[ni]; //the nog node we might kill children at
    //Rcpp::Rcout << "bd: " << "draw proposal" << endl;
    
    //--------------------------------------------------
    //compute things needed for metropolis ratio
    
    double PGny; //prob the nog node grows
    size_t dny = nx->depth();
    PGny = pi.alpha/pow(1.0+dny,pi.beta);
    //Rcpp::Rcout << "[bd]:    PGny = " << PGny << endl;
    
    //better way to code these two?
    double PGlx = pgrow(nx->getl(),xi,pi);
    double PGrx = pgrow(nx->getr(),xi,pi);
    
    double PBy;  //prob of birth move at y
    //if(nx->ntype()=='t') { //is the nog node nx the top node
    if(!(nx->p)) { //is the nog node nx the top node
      PBy = 1.0;
    } else {
      PBy = pi.pb;
    }
    
    double Pboty;  //prob of choosing the nog as bot to split on when y
    int ngood = goodbots.size();
    if(cansplit(nx->getl(),xi)) --ngood; //if can split at left child, lose this one
    if(cansplit(nx->getr(),xi)) --ngood; //if can split at right child, lose this one
    ++ngood;  //know you can split at nx
    Pboty=1.0/ngood;
    
    double PDx = 1.0-PBx; //prob of a death step at x
    double Pnogx = 1.0/nognds.size();
    
    //Rcpp::Rcout << "bd: " << "compute things needed for MH ratio" << endl;
    
    //--------------------------------------------------
    //compute sufficient statistics
    sinfo sl;
    sinfo sr;
    sinfo st;
    getsuff(x, nx->getl(),nx->getr(),xi,di,sl,sr);

    st.n = sl.n + sr.n;
    // Concatentate sl.I and sr.I into st.I
    st.I.clear();
    st.I.reserve(sl.I.size() + sr.I.size());
    st.I.insert(st.I.end(), sl.I.begin(), sl.I.end());
    st.I.insert(st.I.end(), sr.I.begin(), sr.I.end());
    double mu_bar_l = 0.0, mu_bar_r = 0.0, mu_bar_t = 0.0;
    double V_l = 0.0, V_r = 0.0, V_t = 0.0;
    
    mu_posterior_slfm(mu_bar_l, V_l, Phi, sigma, sl, di, pi.sigma_mu[di.d]);
    mu_posterior_slfm(mu_bar_r, V_r, Phi, sigma, sr, di, pi.sigma_mu[di.d]);
    mu_posterior_slfm(mu_bar_t, V_t, Phi, sigma, st, di, pi.sigma_mu[di.d]);
    
    double lill = 0.5 * log(V_l) + 0.5*(mu_bar_l * mu_bar_l)/V_l;
    double lilr = 0.5 * log(V_r) + 0.5*(mu_bar_r * mu_bar_r)/V_r;
    double lilt = 0.5 * log(V_t) + 0.5*(mu_bar_t * mu_bar_t)/V_t;
    //Rcpp::Rcout << "[bd]:    pi.sigma0[" << di.k << "] = " << pi.sigma0[di.k] << endl;
    //Rcpp::Rcout << "[bd]:    st.I.size() = " << st.I.size() << endl;
    //Rcpp::Rcout << "[bd]:    mu_bar_l = " << mu_bar_l << "   V_l = " << V_l << "    lill = " << lill << endl;
    //Rcpp::Rcout << "[bd]:    mu_bar_r = " << mu_bar_r << "   V_r = " << V_r << "    lilr = " << lilr << endl;
    //Rcpp::Rcout << "[bd]:    mu_bar_t = " << mu_bar_t << "   V_t = " << V_t << "    lilt = " << lilt << endl;
    
    double alpha1 = ((1.0-PGny)*PBy*Pboty)/(PGny*(1.0-PGlx)*(1.0-PGrx)*PDx*Pnogx);
    double alpha2 = alpha1*exp(lilt - lill - lilr) * pi.sigma_mu[di.d];
    double alpha = std::min(1.0,alpha2);
    //Rcpp::Rcout << "[bd]:    alpha = " << alpha ;
    //--------------------------------------------------
    //finally ready to try metrop
    unif = gen.uniform();
    //Rcpp::Rcout << "    unif = " << unif << endl;
    if(unif<alpha) {
      // Do the death move. We will set the new mu parameter to be 0.0
      // This will immediately be overwritten in the next step when we call drmu
      //Rcpp::Rcout << " ... death successful" << endl;
      x.death(nx->nid(),0.0);
      //return true;
      return(-1.0*alpha);
    } else {
      return(-1.0*alpha);
    }
  } // closes if/else checking whether it is a birth or death proposal
}
