#include <RcppArmadillo.h>

#include <cmath>
#include "funs.h"
#include "rng.h"
#include <map>
#ifdef MPIBART
#include "mpi.h"
#endif

using Rcpp::Rcout;
using namespace arma;
using namespace Rcpp;




//-------------------------------------------------------------
// FUNCTION: 	Generates realizations from multivariate normal.
//-------------------------------------------------------------
// INPUTS:	   n = sample size
//				   mu = vector of means
//				   sigma = covariance matrix
//-------------------------------------------------------------
// OUTPUT:	n realizations of the specified MVN.
//-------------------------------------------------------------
mat rmvnormArma(int n, vec mu, mat sigma) {

   int ncols = sigma.n_cols;
   mat Y = randn(n, ncols);
   mat result = (repmat(mu, 1, n).t() + Y * chol(sigma)).t();
   return result;
}

//--------------------------------------------------
// normal density N(x, mean, variance)
double pn(double x, double m, double v)
{
	double dif = x-m;
	return exp(-.5*dif*dif/v)/sqrt(2*PI*v);
}

//--------------------------------------------------
// draw from discrete distributin given by p, return index
int rdisc(double *p, RNG& gen)
{

	double sum;
	double u = gen.uniform();

    int i=0;
    sum=p[0];
    while(sum<u) {
		i += 1;
		sum += p[i];
    }
    return i;
}

//--------------------------------------------------
//evalute tree tr on grid given by xi and write to os
void grm(tree& tr, xinfo& xi, std::ostream& os)
{
	size_t p = xi.size();
	if(p!=2) {
		cout << "error in grm, p !=2\n";
		return;
	}
	size_t n1 = xi[0].size();
	size_t n2 = xi[1].size();
	tree::tree_cp bp; //pointer to bottom node
	double *x = new double[2];
	for(size_t i=0;i!=n1;i++) {
		for(size_t j=0;j!=n2;j++) {
			x[0] = xi[0][i];
			x[1] = xi[1][j];
			bp = tr.bn(x,xi);
			os << x[0] << " " << x[1] << " " << bp->getm() << " " << bp->nid() << endl;
		}
	}
	delete[] x;
}

//--------------------------------------------------
//does this bottom node n have any variables it can split on.
bool cansplit(tree::tree_p n, xinfo& xi)
{
	int L,U;
	bool v_found = false; //have you found a variable you can split on
	size_t v=0;
	while(!v_found && (v < xi.size())) { //invar: splitvar not found, vars left
		L=0; U = xi[v].size()-1;
		n->rg(v,&L,&U);
		if(U>=L) v_found=true;
		v++;
	}
	return v_found;
}

//--------------------------------------------------
//compute prob of a birth, goodbots will contain all the good bottom nodes
double getpb(tree &t, xinfo &xi, pinfo &pi, tree::npv &goodbots)
{
	double pb;  //prob of birth to be returned
	tree::npv bnv; //all the bottom nodes
	t.getbots(bnv);
	for(size_t i=0;i!=bnv.size();i++)
		if(cansplit(bnv[i],xi)) goodbots.push_back(bnv[i]);
	if(goodbots.size()==0) { //are there any bottom nodes you can split on?
		pb=0.0;
	} else {
		if(t.treesize()==1) pb=1.0; //is there just one node?
		else pb=pi.pb;
	}
	return pb;
}

double getpb(tree &t, xinfo &xi, pinfo_slfm &pi, tree::npv &goodbots)
{
  double pb;  //prob of birth to be returned
  tree::npv bnv; //all the bottom nodes
  t.getbots(bnv);
  for(size_t i=0;i!=bnv.size();i++)
    if(cansplit(bnv[i],xi)) goodbots.push_back(bnv[i]);
  if(goodbots.size()==0) { //are there any bottom nodes you can split on?
    pb=0.0;
  } else {
    if(t.treesize()==1) pb=1.0; //is there just one node?
    else pb=pi.pb;
  }
  return pb;
}
// overloaded for new prior info classes
double getpb(tree &t, xinfo &xi, tree_prior_info &tree_pi, tree::npv &goodbots){
  double pb; // prob of birth to be returned
  tree::npv bnv; // all the bottom nodes
  t.getbots(bnv); // actually find all of the bottom nodes
  for(size_t i = 0; i != bnv.size(); i++){
    if(cansplit(bnv[i], xi)) goodbots.push_back(bnv[i]);
  }
  if(goodbots.size() == 0) pb = 0.0; // there are no bottom nodes you can split on
  else{
    if(t.treesize() == 1) pb = 1.0; // tree only has one node
    else pb = tree_pi.pb;
  }
  return pb;
}


//--------------------------------------------------
//find variables n can split on, put their indices in goodvars
void getgoodvars(tree::tree_p n, xinfo& xi,  std::vector<size_t>& goodvars)
{
	int L,U;
	for(size_t v=0;v!=xi.size();v++) {//try each variable
		L=0; U = xi[v].size()-1;
		n->rg(v,&L,&U);
		if(U>=L) goodvars.push_back(v);
	}
}

//--------------------------------------------------
//get prob a node grows, 0 if no good vars, else alpha/(1+d)^beta
double pgrow(tree::tree_p n, xinfo &xi, pinfo &pi)
{
	if(cansplit(n,xi)) {
		return pi.alpha/pow(1.0+n->depth(),pi.beta);
	} else {
		return 0.0;
	}
}

double pgrow(tree::tree_p n, xinfo &xi, pinfo_slfm &pi)
{
  if(cansplit(n,xi)) {
    return pi.alpha/pow(1.0+n->depth(),pi.beta);
  } else {
    return 0.0;
  }
}

double pgrow(tree::tree_p n, xinfo &xi, tree_prior_info &tree_pi)
{
  if(cansplit(n,xi)) return tree_pi.alpha/pow(1.0 + n->depth(), tree_pi.beta);
  else return 0.0;
}


void prepare_y(arma::mat &Y, arma::vec &y_col_mean, arma::vec &y_col_sd, arma::vec &y_col_max, arma::vec &y_col_min)
{
  int n_obs = Y.n_rows;
  int q = Y.n_cols;
  double tmp_sum = 0.0; // holds running sum of Y
  double tmp_sum2 = 0.0; // holds running sum of Y^2
  size_t tmp_count = 0; // counts number of missing observations for each outcome
  size_t min_index = -1; // holds index of max element
  size_t max_index = -1; // holds index of min element
  
  // re-size containers
  y_col_mean.set_size(q);
  y_col_sd.set_size(q);
  
  y_col_max.set_size(q);
  y_col_min.set_size(q);
  
  for(size_t k = 0; k < q; k++){
    tmp_sum = 0.0;
    tmp_sum2 = 0.0;
    tmp_count = 0;
    min_index = -1; // set to non-sensical value and then we can initialize
    max_index = -1;
    for(size_t i = 0; i < n_obs; i++){
      if(Y(i,k) == Y(i,k)){
        tmp_count++;
        tmp_sum += Y(i,k);
        tmp_sum2 += Y(i,k) * Y(i,k);
        if(max_index == -1) max_index = i;
        else if(Y(i,k) > Y(max_index,k)) max_index = i;
        
        if(min_index == -1) min_index = i;
        else if(Y(min_index,k) > Y(i,k)) min_index = i;
      } // closes if checking that Y(i,k) is observed
    } // closes loop over observations
    if(tmp_count < 2){ // there is a column of Y which is totally unobserved or only has one observation
      Rcpp::stop("Must have at least two observations per task!");
    } else{
      y_col_mean(k) = tmp_sum/tmp_count;
      y_col_sd(k) = sqrt(1.0/(tmp_count - 1) * (tmp_sum2 - tmp_sum*tmp_sum/tmp_count));
      //y_col_max(k) = Y(max_index,k);
      //y_col_min(k) = Y(min_index,k);
      // We can now re-center and re-scale the data in Y.col(k)
      for(size_t i = 0; i < n_obs; i++){
        if(Y(i,k) == Y(i,k)){
          Y(i,k) -= y_col_mean(k);
          Y(i,k) /= y_col_sd(k);
        }
      } // closes loop over observations
      
      // MUST RE-SCALE/RE-CENTER Y before extracting the min and max of each column.
      // Otherwise the value of sigma_mu will be totally screwed up
      y_col_max(k) = Y(max_index,k);
      y_col_min(k) = Y(min_index,k);
      
    } // closes else checking that we have at least 2 observations of this task
  } // closes loop over tasks
}



// pre-process the observed data y
void prepare_y(arma::mat &Y, std::vector<double> &y_col_mean, std::vector<double> &y_col_sd, std::vector<double> &y_col_max, std::vector<double> &y_col_min)
{
  int n_obs = Y.n_rows;
  int q = Y.n_cols;
  double tmp_sum = 0.0; // holds running sum of Y
  double tmp_sum2 = 0.0; // holds running sum of Y^2
  size_t tmp_count = 0; // counts number of missing observations for each outcome
  size_t min_index = -1; // holds index of max element
  size_t max_index = -1; // holds index of min element
  
  // re-size containers
  y_col_mean.clear();
  y_col_mean.resize(q);

  y_col_sd.clear();
  y_col_sd.resize(q);
  
  y_col_max.clear();
  y_col_max.resize(q);
  
  y_col_min.clear();
  y_col_min.resize(q);
  
  for(size_t k = 0; k < q; k++){
    tmp_sum = 0.0;
    tmp_sum2 = 0.0;
    tmp_count = 0;
    min_index = -1; // set to non-sensical value and then we can initialize
    max_index = -1;
    for(size_t i = 0; i < n_obs; i++){
      if(Y(i,k) == Y(i,k)){
        tmp_count++;
        tmp_sum += Y(i,k);
        tmp_sum2 += Y(i,k) * Y(i,k);
        if(max_index == -1) max_index = i;
        else if(Y(i,k) > Y(max_index,k)) max_index = i;
        
        if(min_index == -1) min_index = i;
        else if(Y(min_index,k) > Y(i,k)) min_index = i;
      } // closes if checking that Y(i,k) is observed
    } // closes loop over observations
    if(tmp_count < 2){ // there is a column of Y which is totally unobserved or only has one observation
      Rcpp::stop("Must have at least two observations per task!");
    } else{
      y_col_mean[k] = tmp_sum/tmp_count;
      y_col_sd[k] = sqrt(1.0/(tmp_count - 1) * (tmp_sum2 - tmp_sum*tmp_sum/tmp_count));
      //y_col_max[k] = Y(max_index,k);
      //y_col_min[k] = Y(min_index,k);
      // We can now re-center and re-scale the data in Y.col(k)
      for(size_t i = 0; i < n_obs; i++){
        if(Y(i,k) == Y(i,k)){
          Y(i,k) -= y_col_mean[k];
          Y(i,k) /= y_col_sd[k];
        }
      } // closes loop over observations
      // MUST RE-SCALE Y before extracting min and max of each column. Otherwise sigma_mu is too large
      y_col_max[k] = Y(max_index,k);
      y_col_min[k] = Y(min_index,k);
    } // closes else checking that we have at least 2 observations of this task
  } // closes loop over tasks
}


// overloaded version of prepare_y meant specifically for the univariate settting
void prepare_y(arma::vec &Y, double &y_mean, double &y_sd, double &y_max, double &y_min)
{
  int n_obs = Y.size();
  double tmp_sum = 0.0; // holds running sum of Y
  double tmp_sum2 = 0.0; // holds running sum of Y^2
  size_t tmp_count = 0; // counts number of missing observations for each outcome
  size_t min_index = -1; // holds index of max element
  size_t max_index = -1; // holds index of min element
  for(size_t i = 0; i < n_obs; i++){
    if(Y(i) == Y(i)){
      tmp_count++;
      tmp_sum += Y(i);
      tmp_sum2 += Y(i) * Y(i);
      if(max_index == -1) max_index = i;
      else if(Y(i) > Y(max_index)) max_index = i;
      
      if(min_index == -1) min_index = i;
      else if(Y(i) < Y(min_index)) min_index = i;
    } // closes if checking that Y(i) is observed
  } // closes loop over the observations
  if(tmp_count < 2) Rcpp::stop("Must have at least two observations");
  else{
    y_mean = tmp_sum/tmp_count;
    y_sd = sqrt(1.0/(tmp_count - 1) * (tmp_sum2 - tmp_sum * tmp_sum/tmp_count));
    //y_min = Y(min_index);
    //y_max = Y(max_index);
    
    // center and scale the data now
    for(size_t i = 0; i < n_obs; i++){
      if(Y(i) == Y(i)){
        Y(i) -= y_mean;
        Y(i) /= y_sd;
      }
    } // closes loop over the observations
    // MUST RESCALE Y before extracting min and max
    y_min = Y(min_index);
    y_max = Y(max_index);
  } // closes else checking that we have at least 2 observed values of Y
  
}


//--------------------------------------------------
//get sufficients stats for all bottom nodes (sy, sy2)
void allsuff(tree& x, xinfo& xi, dinfo& di, tree::npv& bnv, std::vector<sinfo>& sv)
{
  // Bottom nodes are written to bnv.
  // Suff stats for each bottom node are written to elements (each of class sinfo) of sv.
  // Initialize data structures
  tree::tree_cp tbn; //the pointer to the bottom node for the current observations.  tree_cp bc not modifying tree directly.
  size_t ni;         //the  index into vector of the current bottom node
  double *xx;        //current x

  bnv.clear();      // Clear the bnv variable if any value is already saved there.
  x.getbots(bnv);   // Save bottom nodes for x to bnv variable.


  typedef tree::npv::size_type bvsz;  // Is a better C way to set type.  (tree::npv::size_type) will resolve to an integer,
   // or long int, etc.  We don't have to know that ahead of time by using this notation.
  bvsz nb = bnv.size();   // Initialize new var nb of type bvsz for number of bottom nodes, then...
  sv.resize(nb);          // Re-sizing suff stat vector to have same size as bottom nodes.
  
  // need to re-size members of each element of sv
  for(size_t i = 0; i != bnv.size(); i++){
    sv[i].n = 0;
    sv[i].I.clear();
  }

  // bnmap is a tuple (lookups, like in Python).  Want to index by bottom nodes.
  std::map<tree::tree_cp,size_t> bnmap;
  for(bvsz i=0;i!=bnv.size();i++) bnmap[bnv[i]]=i;  // bnv[i]
  //map looks like
  // bottom node 1 ------ 1
  // bottom node 2 ------ 2


  // Loop through each observation.  Push each obs x down the tree and find its bottom node,
  // then index into the suff stat for the bottom node corresponding to that obs.
  
  for(size_t i=0;i<di.n;i++) {
    xx = di.x + i*di.p;  //Index value: di.x is pointer to first element of n*p data vector.  Iterates through each element.
    tbn = x.bn(xx,xi); // Finds bottom node for this observation
    ni = bnmap[tbn]; // Maps bottom node to integer index
    ++(sv[ni].n); // Increment count at this node
    sv[ni].I.push_back(i);
  }

}

void allsuff(tree& x, xinfo& xi, dinfo_slfm& di, tree::npv& bnv, std::vector<sinfo>& sv)
{
  // Bottom nodes are written to bnv.
  // Suff stats for each bottom node are written to elements (each of class sinfo) of sv.
  // Initialize data structures
  tree::tree_cp tbn; //the pointer to the bottom node for the current observations.  tree_cp bc not modifying tree directly.
  size_t ni;         //the  index into vector of the current bottom node
  double *xx;        //current x
  
  bnv.clear();      // Clear the bnv variable if any value is already saved there.
  x.getbots(bnv);   // Save bottom nodes for x to bnv variable.
  
  
  typedef tree::npv::size_type bvsz;  // Is a better C way to set type.  (tree::npv::size_type) will resolve to an integer,
  // or long int, etc.  We don't have to know that ahead of time by using this notation.
  bvsz nb = bnv.size();   // Initialize new var nb of type bvsz for number of bottom nodes, then...
  sv.resize(nb);          // Re-sizing suff stat vector to have same size as bottom nodes.
  
  // need to re-size members of each element of sv
  for(size_t i = 0; i != bnv.size(); i++){
    sv[i].n = 0;
    sv[i].I.clear();
  }
  
  // bnmap is a tuple (lookups, like in Python).  Want to index by bottom nodes.
  std::map<tree::tree_cp,size_t> bnmap;
  for(bvsz i=0;i!=bnv.size();i++) bnmap[bnv[i]]=i;  // bnv[i]
  //map looks like
  // bottom node 1 ------ 1
  // bottom node 2 ------ 2
  
  
  // Loop through each observation.  Push each obs x down the tree and find its bottom node,
  // then index into the suff stat for the bottom node corresponding to that obs.
  
  for(size_t i=0;i<di.n;i++) {
    xx = di.x + i*di.p;  //Index value: di.x is pointer to first element of n*p data vector.  Iterates through each element.
    tbn = x.bn(xx,xi); // Finds bottom node for this observation
    ni = bnmap[tbn]; // Maps bottom node to integer index
    ++(sv[ni].n); // Increment count at this node
    sv[ni].I.push_back(i);
  }
  
}

void allsuff(tree &x, xinfo &xi, data_info &di, tree::npv &bnv, std::vector<sinfo> &sv)
{
  // Bottom nodes are written to bnv.
  // Suff stats for each bottom node are written to elements (each of class sinfo) of sv.
  // Initialize data structures
  tree::tree_cp tbn; //the pointer to the bottom node for the current observations.  tree_cp bc not modifying tree directly.
  size_t ni; //the  index into vector of the current bottom node
  double *xx; //current x
  bnv.clear(); // Clear the bnv variable if any value is already saved there.
  x.getbots(bnv); // Save bottom nodes for x to bnv variable.
  
  
  typedef tree::npv::size_type bvsz;  // Is a better C way to set type.  (tree::npv::size_type) will resolve to an integer,
  // or long int, etc.  We don't have to know that ahead of time by using this notation.
  bvsz nb = bnv.size();   // Initialize new var nb of type bvsz for number of bottom nodes, then...
  sv.resize(nb);          // Re-sizing suff stat vector to have same size as bottom nodes.
  
  // need to re-size members of each element of sv
  for(size_t i = 0; i != bnv.size(); i++){
    sv[i].n = 0;
    sv[i].I.clear();
  }
  
  // bnmap is a tuple (lookups, like in Python).  Want to index by bottom nodes.
  std::map<tree::tree_cp,size_t> bnmap;
  for(bvsz i=0;i!=bnv.size();i++) bnmap[bnv[i]]=i;  // bnv[i]
  //map looks like
  // bottom node 1 ------ 1
  // bottom node 2 ------ 2
  // Loop through each observation.  Push each obs x down the tree and find its bottom node,
  // then index into the suff stat for the bottom node corresponding to that obs.
  
  for(size_t i=0;i<di.n;i++) {
    xx = di.x + i*di.p;  //Index value: di.x is pointer to first element of n*p data vector.  Iterates through each element.
    tbn = x.bn(xx,xi); // Finds bottom node for this observation
    ni = bnmap[tbn]; // Maps bottom node to integer index
    ++(sv[ni].n); // Increment count at this node
    sv[ni].I.push_back(i);
  }
}

//get counts for all bottom nodes
std::vector<int> counts(tree& x, xinfo& xi, dinfo& di, tree::npv& bnv)
{
  tree::tree_cp tbn; //the pointer to the bottom node for the current observations
	size_t ni;         //the  index into vector of the current bottom node
	double *xx;        //current x

	bnv.clear();
	x.getbots(bnv);

	typedef tree::npv::size_type bvsz;
//	bvsz nb = bnv.size();

  std::vector<int> cts(bnv.size(), 0);

	std::map<tree::tree_cp,size_t> bnmap;
	for(bvsz i=0;i!=bnv.size();i++) bnmap[bnv[i]]=i;

	for(size_t i=0;i<di.n;i++) {
		xx = di.x + i*di.p;
		//y=di.y[i];
    //y = di.y[i + k*di.n];

		tbn = x.bn(xx,xi);
		ni = bnmap[tbn];

    cts[ni] += 1;
	}
  return(cts);
}

std::vector<int> counts(tree &x, xinfo &xi, dinfo_slfm &di, tree::npv &bnv)
{
  tree::tree_cp tbn; //the pointer to the bottom node for the current observations
  size_t ni;         //the  index into vector of the current bottom node
  double *xx;        //current x
  
  bnv.clear();
  x.getbots(bnv);
  
  typedef tree::npv::size_type bvsz;
  //  bvsz nb = bnv.size();
  
  std::vector<int> cts(bnv.size(), 0);
  
  std::map<tree::tree_cp,size_t> bnmap;
  for(bvsz i=0;i!=bnv.size();i++) bnmap[bnv[i]]=i;
  
  for(size_t i=0;i<di.n;i++) {
    xx = di.x + i*di.p;
    //y=di.y[i];
    //y = di.y[i + k*di.n];
    
    tbn = x.bn(xx,xi);
    ni = bnmap[tbn];
    
    cts[ni] += 1;
  }
  return(cts);
}

//overloaded for the new prior info classes
std::vector<int> counts(tree &x, xinfo &xi, data_info &di, tree::npv &bnv)
{
  tree::tree_cp tbn; //the pointer to the bottom node for the current observations
  size_t ni;         //the  index into vector of the current bottom node
  double *xx;        //current x
  
  bnv.clear();
  x.getbots(bnv);
  
  typedef tree::npv::size_type bvsz;
  //  bvsz nb = bnv.size();
  
  std::vector<int> cts(bnv.size(), 0);
  
  std::map<tree::tree_cp,size_t> bnmap;
  for(bvsz i=0;i!=bnv.size();i++) bnmap[bnv[i]]=i;
  
  for(size_t i=0;i<di.n;i++) {
    xx = di.x + i*di.p;
    //y=di.y[i];
    //y = di.y[i + k*di.n];
    
    tbn = x.bn(xx,xi);
    ni = bnmap[tbn];
    
    cts[ni] += 1;
  }
  return(cts);
}

void update_counts(int i, std::vector<int> &cts, tree &x, xinfo &xi,
                   dinfo &di,
                   tree::npv &bnv, //vector of pointers to bottom nodes
                   int sign)
{
  tree::tree_cp tbn; //the pointer to the bottom node for the current observations
  size_t ni;         //the  index into vector of the current bottom node
	double *xx;        //current x
	//double y;          //current y

	typedef tree::npv::size_type bvsz;
//	bvsz nb = bnv.size();

	std::map<tree::tree_cp,size_t> bnmap;
	for(bvsz ii=0;ii!=bnv.size();ii++) bnmap[bnv[ii]]=ii; // bnmap[pointer] gives linear index

	xx = di.x + i*di.p;
	//y=di.y[i];
  //y = di.y[i + k*di.n];

	tbn = x.bn(xx,xi);
	ni = bnmap[tbn];

  cts[ni] += sign;
}

void update_counts(int i, std::vector<int> &cts, tree &x, xinfo &xi,
                   dinfo_slfm &di,
                   tree::npv& bnv, //vector of pointers to bottom nodes
                   int sign)
{
  tree::tree_cp tbn; //the pointer to the bottom node for the current observations
  size_t ni;         //the  index into vector of the current bottom node
  double *xx;        //current x
                     //double y;          //current y
  
  typedef tree::npv::size_type bvsz;
  //  bvsz nb = bnv.size();
  
  std::map<tree::tree_cp,size_t> bnmap;
  for(bvsz ii=0;ii!=bnv.size();ii++) bnmap[bnv[ii]]=ii; // bnmap[pointer] gives linear index
  
  xx = di.x + i*di.p;
  //y=di.y[i];
  //y = di.y[i + k*di.n];
  
  tbn = x.bn(xx,xi);
  ni = bnmap[tbn];
  
  cts[ni] += sign;
}

void update_counts(int i, std::vector<int> &cts, tree &x, xinfo &xi,
                   data_info &di,
                   tree::npv &bnv, //vector of pointers to bottom nodes
                   int sign)
{
  tree::tree_cp tbn; //the pointer to the bottom node for the current observations
  size_t ni;         //the  index into vector of the current bottom node
  double *xx;        //current x
  //double y;          //current y
  
  typedef tree::npv::size_type bvsz;
  //  bvsz nb = bnv.size();
  
  std::map<tree::tree_cp,size_t> bnmap;
  for(bvsz ii=0;ii!=bnv.size();ii++) bnmap[bnv[ii]]=ii; // bnmap[pointer] gives linear index
  
  xx = di.x + i*di.p;
  //y=di.y[i];
  //y = di.y[i + k*di.n];
  
  tbn = x.bn(xx,xi);
  ni = bnmap[tbn];
  
  cts[ni] += sign;
}


void update_counts(int i, std::vector<int>& cts, tree& x, xinfo& xi,
                   dinfo& di,
                   std::map<tree::tree_cp,size_t>& bnmap,
                   int sign)
{
  tree::tree_cp tbn; //the pointer to the bottom node for the current observations
  size_t ni;         //the  index into vector of the current bottom node
  double *xx;        //current x
	//double y;          //current y
  /*
	typedef tree::npv::size_type bvsz;
	bvsz nb = bnv.size();

	std::map<tree::tree_cp,size_t> bnmap;
	for(bvsz ii=0;ii!=bnv.size();ii++) bnmap[bnv[ii]]=ii; // bnmap[pointer] gives linear index
	*/
	xx = di.x + i*di.p;
	tbn = x.bn(xx,xi);
	ni = bnmap[tbn];
  cts[ni] += sign;
}

void update_counts(int i, std::vector<int>& cts, tree& x, xinfo& xi,
                   dinfo_slfm& di,
                   std::map<tree::tree_cp,size_t>& bnmap,
                   int sign)
{
  tree::tree_cp tbn; //the pointer to the bottom node for the current observations
  size_t ni;         //the  index into vector of the current bottom node
  double *xx;        //current x
                     //double y;          //current y
  /*
   typedef tree::npv::size_type bvsz;
   bvsz nb = bnv.size();
   
   std::map<tree::tree_cp,size_t> bnmap;
   for(bvsz ii=0;ii!=bnv.size();ii++) bnmap[bnv[ii]]=ii; // bnmap[pointer] gives linear index
   */
  xx = di.x + i*di.p;
  tbn = x.bn(xx,xi);
  ni = bnmap[tbn];
  cts[ni] += sign;
}

void update_counts(int i, std::vector<int>& cts, tree& x, xinfo& xi,
                   data_info& di,
                   std::map<tree::tree_cp,size_t>& bnmap,
                   int sign)
{
  tree::tree_cp tbn; //the pointer to the bottom node for the current observations
  size_t ni;         //the  index into vector of the current bottom node
  double *xx;        //current x
  //double y;          //current y
  /*
   typedef tree::npv::size_type bvsz;
   bvsz nb = bnv.size();
   
   std::map<tree::tree_cp,size_t> bnmap;
   for(bvsz ii=0;ii!=bnv.size();ii++) bnmap[bnv[ii]]=ii; // bnmap[pointer] gives linear index
   */
  xx = di.x + i*di.p;
  tbn = x.bn(xx,xi);
  ni = bnmap[tbn];
  cts[ni] += sign;
}

void update_counts(int i, std::vector<int>& cts, tree& x, xinfo& xi,
                   dinfo& di,
                   std::map<tree::tree_cp,size_t>& bnmap,
                   int sign,
                   tree::tree_cp &tbn
                   )
{
  //tree::tree_cp tbn; //the pointer to the bottom node for the current observations
  size_t ni;         //the  index into vector of the current bottom node
  double *xx;        //current x
  //double y;          //current y
  /*
	typedef tree::npv::size_type bvsz;
	bvsz nb = bnv.size();

	std::map<tree::tree_cp,size_t> bnmap;
	for(bvsz ii=0;ii!=bnv.size();ii++) bnmap[bnv[ii]]=ii; // bnmap[pointer] gives linear index
	*/
	xx = di.x + i*di.p;
	tbn = x.bn(xx,xi);
	ni = bnmap[tbn];

  cts[ni] += sign;
}
void update_counts(int i, std::vector<int>& cts, tree& x, xinfo& xi,
                   dinfo_slfm& di,
                   std::map<tree::tree_cp,size_t>& bnmap,
                   int sign,
                   tree::tree_cp &tbn
                   )
{
  //tree::tree_cp tbn; //the pointer to the bottom node for the current observations
  size_t ni;         //the  index into vector of the current bottom node
  double *xx;        //current x
                     //double y;          //current y
  /*
   typedef tree::npv::size_type bvsz;
   bvsz nb = bnv.size();
   
   std::map<tree::tree_cp,size_t> bnmap;
   for(bvsz ii=0;ii!=bnv.size();ii++) bnmap[bnv[ii]]=ii; // bnmap[pointer] gives linear index
   */
  xx = di.x + i*di.p;
  tbn = x.bn(xx,xi);
  ni = bnmap[tbn];
  
  cts[ni] += sign;
}

void update_counts(int i, std::vector<int>& cts, tree& x, xinfo& xi,
                   data_info& di,
                   std::map<tree::tree_cp,size_t>& bnmap,
                   int sign,
                   tree::tree_cp &tbn
                   )
{
  //tree::tree_cp tbn; //the pointer to the bottom node for the current observations
  size_t ni;         //the  index into vector of the current bottom node
  double *xx;        //current x
  //double y;          //current y
  /*
   typedef tree::npv::size_type bvsz;
   bvsz nb = bnv.size();
   
   std::map<tree::tree_cp,size_t> bnmap;
   for(bvsz ii=0;ii!=bnv.size();ii++) bnmap[bnv[ii]]=ii; // bnmap[pointer] gives linear index
   */
  xx = di.x + i*di.p;
  tbn = x.bn(xx,xi);
  ni = bnmap[tbn];
  
  cts[ni] += sign;
}

// check minimum leaf size
bool min_leaf(int minct, std::vector<tree>& t, xinfo& xi, dinfo& di) {
  bool good = true;
  tree::npv bnv;
  std::vector<int> cts;
  int m = 0;
  for (size_t tt=0; tt<t.size(); ++tt) {
    cts = counts(t[tt], xi, di, bnv);
    m = std::min(m, *std::min_element(cts.begin(), cts.end()));
    if(m<minct) {
      good = false;
      break;
    }
  }
  return good;
}
bool min_leaf(int minct, std::vector<tree>& t, xinfo& xi, dinfo_slfm& di) {
  bool good = true;
  tree::npv bnv;
  std::vector<int> cts;
  int m = 0;
  for (size_t tt=0; tt<t.size(); ++tt) {
    cts = counts(t[tt], xi, di, bnv);
    m = std::min(m, *std::min_element(cts.begin(), cts.end()));
    if(m<minct) {
      good = false;
      break;
    }
  }
  return good;
}

bool min_leaf(int minct, std::vector<tree> &t, xinfo &xi, data_info &di) {
  bool good = true;
  tree::npv bnv;
  std::vector<int> cts;
  int m = 0;
  for (size_t tt=0; tt<t.size(); ++tt) {
    cts = counts(t[tt], xi, di, bnv);
    m = std::min(m, *std::min_element(cts.begin(), cts.end()));
    if(m<minct) {
      good = false;
      break;
    }
  }
  return good;
}


//--------------------------------------------------
//get sufficient stats for children (v,c) of node nx in tree x for outcome k
// [SKD]: this is used in the birth proposals
// In birth proposals, we split the node nx in tree x according to cutpoint c of variable v
//

void getsuff(tree &x, tree::tree_cp nx, size_t v, size_t c, xinfo &xi, dinfo &di, sinfo &sl, sinfo &sr)
{
	double *xx;//current x
  sl.n = 0; // counts number of observations in this leaf
  sl.I.clear(); // index of observations in this leaf
  sr.n = 0; // counts number of observations in this leaf
  sr.I.clear();
	for(size_t i=0;i<di.n;i++) {
		xx = di.x + i*di.p; // gets the next x
		if(nx==x.bn(xx,xi)) { //does the bottom node = xx's bottom node. If so, we want to see if it goes to left or right child
			if(xx[v] < xi[v][c]) { // xx goes to the left child
				sl.n++;
        sl.I.push_back(i);
			} else { // xx goes to the right child
				sr.n++;
        sr.I.push_back(i);
			}
		} // closes if checking whether obs i goes to bottom node nx
	} // closes loop over observations
}

void getsuff(tree &x, tree::tree_cp nx, size_t v, size_t c, xinfo &xi, dinfo_slfm &di, sinfo &sl, sinfo &sr)
{
  double *xx;
  sl.n = 0; // counts number of observations in the left child leaf
  sl.I.clear();
  sr.n = 0; // counts number of observations in the right child leaf
  sr.I.clear();
  for(size_t i = 0; i < di.n; i++){
    xx = di.x + i*di.p ; // points to the next x
    if(nx == x.bn(xx,xi)){ // does bottom node = xx's bottom node. If so, we need to see if xx goes to left or right child
      if(xx[v] < xi[v][c]){ // xx goes to the left child
        sl.n++;
        sl.I.push_back(i);
      } else{ // xx goes to the right child
        sr.n++;
        sr.I.push_back(i);
      }
    } // closes if checking whether obs i goes to bottom node nx
  } // closes loop over the observations i
}

void getsuff(tree &x, tree::tree_cp nx, size_t v, size_t c, xinfo &xi, data_info &di, sinfo &sl, sinfo &sr)
{
  double *xx;
  sl.n = 0; // counts number of observations in the left child leaf
  sl.I.clear();
  sr.n = 0; // counts number of observations in the right child leaf
  sr.I.clear();
  for(size_t i = 0; i < di.n; i++){
    xx = di.x + i*di.p ; // points to the next x
    if(nx == x.bn(xx,xi)){ // does bottom node = xx's bottom node. If so, we need to see if xx goes to left or right child
      if(xx[v] < xi[v][c]){ // xx goes to the left child
        sl.n++;
        sl.I.push_back(i);
      } else{ // xx goes to the right child
        sr.n++;
        sr.I.push_back(i);
      }
    } // closes if checking whether obs i goes to bottom node nx
  } // closes loop over the observations i
}


//--------------------------------------------------
//get sufficient stats for pair of bottom children nl(left) and nr(right) in tree x
// [SKD] : This is used in the death proposals
// In death proposal nodes nl and nr in tree x get combined into a single node
void getsuff(tree &x, tree::tree_cp nl, tree::tree_cp nr, xinfo &xi, dinfo &di, sinfo &sl, sinfo &sr)
{
	double *xx;//current x
	//double y;  //current y
	sl.n=0;
  sl.I.clear();
  sr.n = 0;
  sr.I.clear();
	for(size_t i=0;i<di.n;i++) {
		xx = di.x + i*di.p;
		tree::tree_cp bn = x.bn(xx,xi);
		if(bn==nl) {
      sl.n++;
      sl.I.push_back(i); // add i to the index of the node
		} // closes if checking if observaiton goes to left child
		if(bn==nr) {
      sr.n++; // increment count of observations in this node
      sr.I.push_back(i); // add i to the index of the node
		} // closes if checking if observation goes to right child
	} // closes loop over observations
}

void getsuff(tree &x, tree::tree_cp nl, tree::tree_cp nr, xinfo &xi, dinfo_slfm &di, sinfo &sl, sinfo &sr)
{
  double *xx; // current x
  sl.n = 0;
  sl.I.clear();
  sr.n = 0;
  sr.I.clear();
  for(size_t i = 0; i < di.n; i++){
    xx = di.x + i*di.p;
    tree::tree_cp bn = x.bn(xx,xi);
    if(bn == nl){
      sl.n++;
      sl.I.push_back(i) ; // add i to the index of the node for left-child
    }
    if(bn == nr){
      sr.n++;
      sr.I.push_back(i);
    }
  } // closes loop over observations
}

void getsuff(tree &x, tree::tree_cp nl, tree::tree_cp nr, xinfo &xi, data_info &di, sinfo &sl, sinfo &sr)
{
  double *xx;//current x
  //double y;  //current y
  sl.n=0;
  sl.I.clear();
  sr.n = 0;
  sr.I.clear();
  for(size_t i=0;i<di.n;i++) {
    xx = di.x + i*di.p;
    tree::tree_cp bn = x.bn(xx,xi);
    if(bn==nl) {
      sl.n++;
      sl.I.push_back(i); // add i to the index of the node
    } // closes if checking if observaiton goes to left child
    if(bn==nr) {
      sr.n++; // increment count of observations in this node
      sr.I.push_back(i); // add i to the index of the node
    } // closes if checking if observation goes to right child
  } // closes loop over observations
}

// ---------------------------
// compute posterior mean of terminal node parameters mu
/*
void mu_posterior_multi(double &mu_bar, double &V, const arma::mat &Omega, const sinfo &si, dinfo &di, const double sigma_mu)
{
  double* xx;
  double V_inv = 1.0/(sigma_mu * sigma_mu);
  mu_bar = 0.0;
  if(si.n > 0){
    for(size_t i = 0; i < si.I.size(); i++){
      xx = di.x + si.I[i]*di.p; // now points to the x values for observation si.I[i]
      V_inv += Omega(di.k,di.k); // for heteroskedastic this needs to be
      mu_bar += Omega(di.k, di.k) * di.r_p[si.I[i]]; // alternatively could use *(di.r_p + si.I[i]) or some such
      for(size_t kk = 0; kk < di.q; kk++){
        if(kk != di.k) mu_bar += Omega(di.k, kk) * di.r_f[kk + si.I[i] + di.q];
      }
    }
  }
  V = 1.0/V_inv;
  mu_bar *= V;
}
*/




void mu_posterior_uni(double &mu_bar, double &V, const double &omega, const sinfo &si, dinfo &di, const double sigma_mu)
{
  double* xx;
  double V_inv = 1.0/(sigma_mu * sigma_mu);
  mu_bar = 0.0;
  if(si.n > 0){
    for(size_t i = 0; i < si.I.size(); i++){
      xx = di.x + si.I[i]*di.p; // now points to the x values for observation si.I[i]
      V_inv += omega; // for heteroskedastic this needs to a function of x
      mu_bar += omega * di.r_p[si.I[i]];
    }
  }
  V = 1.0/V_inv;
  mu_bar *= V;
}

// overloaded
// The below function was what was being used prior to 17 June 2019
// the way it access di.delta is not portable to the multi-task setting when we want to have task-specific fits

void mu_posterior_uni(double &mu_bar, double &V, const double &sigma, const sinfo &si, data_info &di, tree_prior_info &tree_pi)
{
  double V_inv = 1.0/(tree_pi.sigma_mu * tree_pi.sigma_mu);
  mu_bar = 0.0;
  if(si.n > 0){
    for(size_t i = 0; i < si.I.size(); i++){
      if(di.delta[si.I[i]] == 1){ // we actually observe the outcome
        V_inv += di.weight / (sigma * sigma);
        mu_bar += di.weight * tree_pi.r_p[si.I[i]]/(sigma * sigma);
      }
    }
  }
  V = 1.0/V_inv;
  mu_bar *= V;
}

/* 17 June 2019: mu_posterior_uni currently accesses di.delta[si.I[i]]
  If we want to have separate univariate BART fits for multi-task data, we need to access something slightly different
  We really need to check di.delta[k + si.I[i] * q]
 
 */
void mu_posterior_uni(double &mu_bar, double &V, const double &sigma, const sinfo &si, data_info &di, tree_prior_info &tree_pi, size_t k)
{
  double V_inv = 1.0/(tree_pi.sigma_mu * tree_pi.sigma_mu);
  mu_bar = 0.0;
  if(si.n > 0){
    for(size_t i = 0; i < si.I.size(); i++){
      if(di.delta[k + si.I[i]*di.q] == 1){
        V_inv += di.weight / (sigma * sigma);
        mu_bar += di.weight * tree_pi.r_p[si.I[i]]/(sigma * sigma);
      }
    }
  }
  V = 1.0/V_inv;
  mu_bar *= V;
}



void mu_posterior_slfm(double &mu_bar, double &V, const arma::mat Phi, const arma::vec sigma, sinfo &si, dinfo_slfm &di, double sigma_mu)
{
  double* xx;
  double V_inv = 1.0/(sigma_mu * sigma_mu);
  double r = 0.0; // the residual
  mu_bar = 0.0;
  if(si.n > 0){ // only update if there are observations at the terminal node
    for(size_t i = 0; i < si.I.size(); i++){ // note we need to use si.I[i] in the computations
      //Rcpp::Rcout << "  observation " << si.I[i] << endl;
      xx = di.x + si.I[i]*di.p; // now points to the xvalues for observation si.I[i]. This isn't needed here.
      for(size_t k = 0; k < di.q; k++){
        //Rcpp::Rcout << "    k = " << k ;
        if(di.delta[k + si.I[i]*di.q] == 1){ // we actually observe observation i, task k
          V_inv += Phi(k,di.d) * Phi(k, di.d) /(sigma(k) * sigma(k));
          // note that af currently contains the fit of every tree but tree t in basis function d
          r = di.y[k + si.I[i]*di.q] - di.af[k + si.I[i]*di.q]; // partial residual
          //Rcpp::Rcout << " r = " << r << endl;
          //if(r != r) Rcpp::Rcout << "[mu_posterior_slfm]: nan in r. i = " << i << " k = " << k << endl;
          mu_bar += Phi(k, di.d) * r/(sigma(k) * sigma(k));
        } else{
          //Rcpp::Rcout << " observation missing. skipping this!" << endl;
        } // closes if checking that observation i is available for task k
      }// closes loop over tasks
    } // closes loop over observations
  } // closes if checking that there are observation in terminal node
  V = 1.0/V_inv;
  //if(V != V) Rcpp::Rcout << "[mu_posterior_slfm]: nan in V. V_inv = " << V_inv << endl;
  mu_bar *= V;
}

//overloaded for the new prior info classes
void mu_posterior_slfm(double &mu_bar, double &V, const arma::mat &Phi, const std::vector<double> &sigma, sinfo &si, data_info &di, tree_prior_info &tree_pi, phi_prior_info &phi_pi)
{
  double V_inv = 1.0/(tree_pi.sigma_mu * tree_pi.sigma_mu);
  mu_bar = 0.0;
  if(si.n > 0){ // only update if there are observations at the terminal node
    for(size_t i = 0; i < si.I.size(); i++){
      for(size_t k = 0; k < phi_pi.q; k++){
        if(di.delta[k + si.I[i]*di.q] == 1){
          V_inv += di.weight * Phi(k, phi_pi.d) * Phi(k, phi_pi.d)/(sigma[k] * sigma[k]);
          mu_bar += di.weight * tree_pi.r_p[k + si.I[i]* di.q] * Phi(k, phi_pi.d)/(sigma[k] * sigma[k]);
        }
      } // closes loop over tasks
    } // closes loop over observations
  } // closes if checking that there are observations in terminal node
  
  if(V_inv == 0) Rcpp::Rcout << "[mu_posterior_slfm]: V_inv = 0";
  V = 1.0/V_inv;
  mu_bar *= V;
  /*
  // let's try the alternative formula. If there are discrepancies we will find them here.
  double tmp_V_inv = 1.0/(tree_pi.sigma_mu * tree_pi.sigma_mu);
  double tmp_V = 0.0;
  double tmp_mu_bar = 0.0;
  double r = 0.0;
  if(si.n > 0){ // only update if there are observations at the terminal node
    for(size_t i = 0; i < si.I.size(); i++){ // note we need to use si.I[i] in the computations
      //Rcpp::Rcout << "  observation " << si.I[i] << endl;
      for(size_t k = 0; k < di.q; k++){
        //Rcpp::Rcout << "    k = " << k ;
        if(di.delta[k + si.I[i]*di.q] == 1){ // we actually observe observation i, task k
          tmp_V_inv += Phi(k,phi_pi.d) * Phi(k, phi_pi.d) /(sigma[k] * sigma[k]);
          // note that af currently contains the fit of every tree but tree t in basis function d
          r = di.y[k + si.I[i]*di.q] - di.af[k + si.I[i]*di.q]; // partial residual
          //Rcpp::Rcout << " r = " << r << endl;
          //if(r != r) Rcpp::Rcout << "[mu_posterior_slfm]: nan in r. i = " << i << " k = " << k << endl;
          tmp_mu_bar += Phi(k, phi_pi.d) * r/(sigma[k] * sigma[k]);
        } else{
          //Rcpp::Rcout << " observation missing. skipping this!" << endl;
        } // closes if checking that observation i is available for task k
      }// closes loop over tasks
    } // closes loop over observations
  } // closes if checking that there are observation in terminal node
  tmp_V = 1.0/tmp_V_inv;
  //if(V != V) Rcpp::Rcout << "[mu_posterior_slfm]: nan in V. V_inv = " << V_inv << endl;
  tmp_mu_bar *= tmp_V;
  
  if( (abs(tmp_mu_bar - mu_bar) > 1e-6) || (abs(tmp_V - V) > 1e-6)) {
    Rcpp::Rcout << "[mu_posterior_slfm]: tmp_mu_bar = " << tmp_mu_bar << " mu_bar = " << mu_bar << endl;
    Rcpp::Rcout << "[mu_posterior_slfm]: tmp_V = " << tmp_V << " V = " << V << endl;
  }
  */
}


//--------------------------------------------------
//fit for multiple data points, not by reference.
void fit(tree& t, xinfo& xi, dinfo& di, double* fv)
{
  double *xx;
  tree::tree_cp bn;
  for(size_t i=0;i<di.n;i++) {
    xx = di.x + i*di.p;
    bn = t.bn(xx,xi);
    fv[i] = bn->getm();
  }
}

void fit(tree& t, xinfo& xi, dinfo_slfm& di, double* fv){
  double *xx;
  tree::tree_cp bn;
  for(size_t i = 0; i < di.n;i++){
    xx = di.x + i*di.p;
    bn = t.bn(xx,xi);
    fv[i] = bn->getm();
  }
}
void fit(tree& t, xinfo &xi, data_info &di, double* fv)
{
  double *xx;
  tree::tree_cp bn;
  for(size_t i=0;i<di.n;i++) {
    xx = di.x + i*di.p;
    bn = t.bn(xx,xi);
    fv[i] = bn->getm();
  }
}

//--------------------------------------------------
//partition
void partition(tree& t, xinfo& xi, dinfo& di, std::vector<size_t>& pv)
{
	double *xx;
	tree::tree_cp bn;
	pv.resize(di.n);
	for(size_t i=0;i<di.n;i++) {
		xx = di.x + i*di.p;
		bn = t.bn(xx,xi);
		pv[i] = bn->nid();
	}
}
//--------------------------------------------------
// draw all the bottom node mu's
/*
void drmu_multi(tree &t, const arma::mat &Omega, xinfo &xi, dinfo &di, pinfo &pi, RNG &gen)
{
  tree::npv bnv;
  std::vector<sinfo> sv;
  allsuff(t, xi, di, bnv, sv);
  
  double mu_bar = 0.0;
  double V = 0.0;
  
  for(tree::npv::size_type i = 0; i != bnv.size(); i++){
    mu_bar = 0.0;
    V = 0.0;
    
    mu_posterior_multi(mu_bar, V, Omega, sv[i], di, pi.sigma_mu[di.k]);
    bnv[i]->setm(mu_bar + sqrt(V) * gen.normal());
    if(bnv[i]->getm() != bnv[i]->getm()){
      // something went horribly horribly wrong!
      //for(int ii = 0; ii<di.n;ii++) Rcout << *(di.x + ii*di.p) << " " ;
      Rcpp::Rcout << "mu_bar = " << mu_bar << " V = " << V << endl;
      Rcpp::stop("drmu_failed");
    }
  } // closes loop over the vector of bottom nodes
}
*/
void drmu_uni(tree &t, const double &omega, xinfo &xi, dinfo &di, pinfo &pi, RNG &gen){
  tree::npv bnv;
  std::vector<sinfo> sv;
  allsuff(t, xi, di, bnv, sv);
  
  double mu_bar = 0.0;
  double V = 0.0;
  for(tree::npv::size_type i = 0; i != bnv.size(); i++){
    mu_bar = 0.0;
    V = 0.0;
    
    mu_posterior_uni(mu_bar, V, omega, sv[i], di, pi.sigma_mu[di.k]);
    bnv[i]->setm(mu_bar + sqrt(V) * gen.normal());
    if(bnv[i]->getm() != bnv[i]->getm()){
      // something went horribly horribly wrong!
      //for(int ii = 0; ii<di.n;ii++) Rcout << *(di.x + ii*di.p) << " " ;
      //Rcout << endl << "mu_bar = " << mu_bar << " V = " << V << endl;
      Rcpp::Rcout << "mu_bar = " << mu_bar << " V = " << V << endl;

      Rcpp::stop("drmu_failed");
    }
  } // closes loop over the vector of bottom nodes
}

//overloaded for the new prior info classes

void drmu_uni(tree &t, const double &sigma, xinfo &xi, data_info &di, tree_prior_info &tree_pi, RNG &gen)
{
  tree::npv bnv; // bottom nodes
  std::vector<sinfo> sv; // for sufficient stats for each bottom node
  allsuff(t, xi, di, bnv, sv); // get all sufficient statistics for each bottom node
  
  double mu_bar = 0.0;
  double V = 0.0;
  for(tree::npv::size_type i = 0; i != bnv.size(); i++){
    mu_bar = 0.0;
    V = 0.0;
    mu_posterior_uni(mu_bar, V, sigma, sv[i], di, tree_pi);
    bnv[i]->setm(mu_bar + sqrt(V) * gen.normal());
    if(bnv[i]->getm() != bnv[i]->getm()){
      Rcpp::Rcout << "nan in drmu_uni" << endl;
      Rcpp::Rcout << "problem is in node " << i << " of " << bnv.size() << endl;
      Rcpp::Rcout << "node members : ";
      for(size_t ii = 0; ii < sv[i].n; ii++) Rcpp::Rcout << " " << sv[i].I[ii] ;
      Rcpp::Rcout << endl;
      Rcpp::Rcout << "mu_bar = " << mu_bar << "  V = " << V << endl;
      Rcpp::stop("nan in drmu_uni");
    }
  } // closes loop over the bottom nodes
}

// new function from 17 June that corrects the look-up in di.delta
void drmu_uni(tree &t, const double &sigma, xinfo &xi, data_info &di, tree_prior_info &tree_pi, size_t k, RNG &gen)
{
  tree::npv bnv; // bottom nodes
  std::vector<sinfo> sv; // for sufficient stats for each bottom node
  allsuff(t, xi, di, bnv, sv); // get all sufficient  statistics for each bottom node
  
  double mu_bar = 0.0;
  double V = 0.0;
  for(tree::npv::size_type i = 0; i != bnv.size(); i++){
    mu_bar = 0.0;
    V = 0.0;
    mu_posterior_uni(mu_bar, V, sigma, sv[i], di, tree_pi, k);
    bnv[i]->setm(mu_bar + sqrt(V) * gen.normal());
    if(bnv[i]->getm() != bnv[i]->getm()){
      Rcpp::Rcout << "nan in drmu_uni" << endl;
      Rcpp::Rcout << "problem is in node " << i << " of " << bnv.size() << endl;
      Rcpp::Rcout << "node members : ";
      for(size_t ii = 0; ii < sv[i].n; ii++) Rcpp::Rcout << " " << sv[i].I[ii] ;
      Rcpp::Rcout << endl;
      Rcpp::Rcout << "mu_bar = " << mu_bar << "  V = " << V << endl;
      Rcpp::stop("nan in drmu_uni");
    }
  }
}



void drmu_slfm(tree &t, const arma::mat Phi, const arma::vec sigma, xinfo &xi, dinfo_slfm &di, pinfo_slfm &pi, RNG &gen)
{
  tree::npv bnv;
  std::vector<sinfo> sv;
  allsuff(t, xi, di, bnv, sv);
  
  double mu_bar = 0.0;
  double V = 0.0;
  for(tree::npv::size_type i = 0; i != bnv.size(); i++){
    mu_bar = 0.0;
    V = 0.0;
    mu_posterior_slfm(mu_bar, V, Phi, sigma, sv[i], di, pi.sigma_mu[di.d]);
    //Rcpp::Rcout << "[drmu_slfm]: mu_bar = " << mu_bar << " V = " << V << endl;
    bnv[i]->setm(mu_bar + sqrt(V) * gen.normal());
    if(bnv[i]->getm() != bnv[i]->getm()){
      //for(size_t ii = 0; ii < sv[i].I.size(); ii++) Rcpp::Rcout << " " << ii ;
      //Rcpp::Rcout << endl;
      Rcpp::Rcout << "mu_bar = " << mu_bar << " V = " << V << endl;
      Rcpp::stop("drmu failed: nan in terminal node");
    }
  } // closes loop over the vector of bottom nodes of tree t
}

// overloaded for the new prior information classes
void drmu_slfm(tree &t, const arma::mat &Phi, const std::vector<double> &sigma, xinfo &xi, data_info &di, tree_prior_info &tree_pi, phi_prior_info &phi_pi, RNG &gen)
{
  tree::npv bnv;
  std::vector<sinfo> sv;
  allsuff(t, xi, di, bnv, sv);
  
  double mu_bar = 0.0;
  double V = 0.0;
  for(tree::npv::size_type i = 0; i != bnv.size(); i++){
    mu_bar = 0.0;
    V = 0.0;
    mu_posterior_slfm(mu_bar, V, Phi, sigma, sv[i], di, tree_pi, phi_pi);
    bnv[i]->setm(mu_bar + sqrt(V) * gen.normal());
    if(bnv[i]->getm() != bnv[i]->getm()) Rcpp::stop("drmu failed: nan in terminal node");
  } // closes loop over bottom nodes of tree t
}

//--------------------------------------------------
//write cutpoint information to screen
void prxi(xinfo& xi)
{
	cout << "xinfo: \n";
	for(size_t v=0;v!=xi.size();v++) {
		cout << "v: " << v << endl;
		for(size_t j=0;j!=xi[v].size();j++) cout << "j,xi[v][j]: " << j << ", " << xi[v][j] << endl;
	}
	cout << "\n\n";
}

//--------------------------------------------------
//make xinfo = cutpoints
void makexinfo(size_t p, size_t n, double *x, xinfo& xi, size_t nc)
{
	double xinc;

	//compute min and max for each x
	std::vector<double> minx(p,INFINITY);
	std::vector<double> maxx(p,-INFINITY);
	double xx;
	for(size_t i=0;i<p;i++) {
		for(size_t j=0;j<n;j++) {
			xx = *(x+p*j+i);
			if(xx < minx[i]) minx[i]=xx;
			if(xx > maxx[i]) maxx[i]=xx;
		}
	}
	//make grid of nc cutpoints between min and max for each x.
	xi.resize(p);
	for(size_t i=0;i<p;i++) {
		xinc = (maxx[i]-minx[i])/(nc+1.0);
		xi[i].resize(nc);
		for(size_t j=0;j<nc;j++) xi[i][j] = minx[i] + (j+1)*xinc;
	}
}
// get min/max needed to make cutpoints
void makeminmax(size_t p, size_t n, double *x, std::vector<double> &minx, std::vector<double> &maxx)
{
	double xx;

	for(size_t i=0;i<p;i++) {
		for(size_t j=0;j<n;j++) {
			xx = *(x+p*j+i);
			if(xx < minx[i]) minx[i]=xx;
			if(xx > maxx[i]) maxx[i]=xx;
		}
	}
}
//make xinfo = cutpoints give the minx and maxx vectors
void makexinfominmax(size_t p, xinfo& xi, size_t nc, std::vector<double> &minx, std::vector<double> &maxx)
{
	double xinc;
	//make grid of nc cutpoints between min and max for each x.
	xi.resize(p);
	for(size_t i=0;i<p;i++) {
		xinc = (maxx[i]-minx[i])/(nc+1.0);
		xi[i].resize(nc);
		for(size_t j=0;j<nc;j++) xi[i][j] = minx[i] + (j+1)*xinc;
	}
}

// Check if a vector is sorted.  For checking z and zpred for causal funbart.
bool is_sort(arma::vec x) {
     int n=x.n_elem;
     for (int i=0; i<n-1; ++i)
         if (x[i] < x[i+1]) return false;
     return true;
}

// Function to update Phi
void update_Phi_gaussian(arma::mat &Phi, const arma::vec &sigma, dinfo_slfm &di, pinfo_slfm &pi,  RNG &gen)
{
  
  double r = 0.0; // partial residual. may not need to track the entire array of partial residuals
  double mu_phi = 0.0;
  double V_phi = 0.0;
  double V_phi_inv = 0.0;
  for(int k = 0; k < di.q; k++){
    for(int d = 0; d < di.D; d++){
      mu_phi = 0.0;
      V_phi_inv = 1.0/(pi.sigma_phi[k] * pi.sigma_phi[k]);
      for(int i = 0; i < di.n; i++){
        if(di.delta[k + i*di.q] == 1){ // check that we observe task k for observation i
          r = di.y[k + i*di.q];
          for(int dd = 0; dd < di.D; dd++){
            if(dd != d) r -= Phi(k,dd) * di.uf[dd + i * di.D];
          }
          // at this point r contains the current partial residual based on the fit all other (d-1) basis functions.
          // note that we cannot use allfit here because it is based on the previous version of Phi
          //mu_phi += r/(sigma(k) * sigma(k)); // wrong formula! It is missing a factor of u_d(x_i) .. caught on 20 March 2019
          mu_phi += r/(sigma(k) * sigma(k)) * di.uf[d + i * di.D];
          
          V_phi_inv += (di.uf[d + i*di.D] * di.uf[d + i*di.D])/(sigma(k) * sigma(k));
        } // closes if checking that we observe task k for observation i
      } // closes loop over the observations i
      V_phi = 1.0/V_phi_inv;
      mu_phi *= V_phi;
      Phi(k,d) = mu_phi + sqrt(V_phi) * gen.normal(); // draw from the appropriate normal distribution
    } // closes loop over the basis functions
  } // closes loop over the tasks
  
}

void update_Phi_gaussian(arma::mat &Phi, const std::vector<double> &sigma, data_info &di, phi_prior_info &phi_pi, RNG &gen)
{
  double r = 0.0; // partial residual
  double mu_phi = 0.0;
  double V_phi = 0.0;
  double V_phi_inv = 0.0;
  for(size_t k = 0; k < di.q; k++){
    for(size_t d = 0; d < phi_pi.D; d++){
      mu_phi = 0.0;
      V_phi_inv = 1.0/(phi_pi.sigma_phi[k] * phi_pi.sigma_phi[k]);
      for(size_t i = 0; i < di.n; i++){
        if(di.delta[k + i*di.q] == 1){
          r = di.y[k + i*di.q];
          for(size_t dd = 0; dd < phi_pi.D; dd++){
            if(dd != d) r -= Phi(k,dd) * phi_pi.uf[dd + i * phi_pi.D];
          }
          // r contains current partial residuals based on fit of all other (d-1) basis functions
          mu_phi += di.weight * r * phi_pi.uf[d + i* phi_pi.D]/(sigma[k] * sigma[k]);
          V_phi_inv += (di.weight * phi_pi.uf[d + i * phi_pi.D] * phi_pi.uf[d + i*phi_pi.D])/(sigma[k] * sigma[k]);
        } // closes if checking that we observe task k observation i
      } // closes loop over observations
      V_phi = 1.0/V_phi_inv;
      mu_phi *= V_phi;
      Phi(k,d) = mu_phi + sqrt(V_phi) * gen.normal(); // draw Phi(k,d) from the appropriate normal prior
    } // closes loop over basis functions d
  } // closes loop over tasks k
}




void update_Phi_ss(arma::mat &Phi, arma::vec &theta, const arma::vec &sigma, dinfo_slfm &di, pinfo_slfm &pi,  RNG &gen)
{
  double r = 0.0; // partial residual
  double mu_phi_0 = 0.0; // posterior mean when gamma = 0... this is always 0 because prior mean of phi is 0
  double mu_phi_1 = 0.0; // posterior mean when gamma = 1... this is basically mu_phi from update_Phi_gaussian
  double v_phi_0_inv = 0.0; // posterior precision when gamma = 0 ... alway 1/(pi.sigma_phi[k] * pi.sigma_phi[k])
  double v_phi_1_inv = 0.0; // posterior precision when gamma = 1
  double v_phi_0 = 0.0;
  double v_phi_1 = 0.0;
  double log_p0 = 0.0;
  double log_p1 = 0.0;
  double p1 = 0.0;
  double p0 = 0.0;
  int gamma_sum = 0; // counts number of gammas that are bigger than 0
  
  for(size_t d = 0; d < di.D; d++){
    gamma_sum = 0; // counts how many gamma_k,d's are active for basis function d
    for(size_t k = 0; k < di.q; k++){
      mu_phi_0 = 0.0;
      mu_phi_1 = 0.0;
      v_phi_0_inv = 1.0/(pi.sigma_phi[k] * pi.sigma_phi[k]);
      v_phi_1_inv = 1.0/(pi.sigma_phi[k] * pi.sigma_phi[k]);
      for(size_t i = 0; i < di.n; i++){
        if(di.delta[k + i*di.q] == 1){
          r = di.y[k + i*di.q];
          for(size_t dd = 0; dd < di.D; dd++){
            if(dd != d) r -= Phi(k,dd) * di.uf[dd + i*di.D];
          }
          // at this point r contains the current partial residual based on the fit all other (d-1) basis functions
          mu_phi_1 += r/(sigma(k) * sigma(k)) * di.uf[d + i*di.D];
          v_phi_1_inv += (di.uf[d + i*di.D] * di.uf[d + i*di.D])/(sigma(k) * sigma(k));
        } // check that observe task k for observation i
      } // closes loop over the observations i
      v_phi_0 = 1.0/v_phi_0_inv;
      v_phi_1 = 1.0/v_phi_1_inv;
      mu_phi_0 *= v_phi_0;
      mu_phi_1 *= v_phi_1;
      
      // more useful to work on the log-scale for the probabilities
      
      log_p0 = log(1.0 - theta(d)) + 1/2 * log(v_phi_0) + 1/2 * mu_phi_0 * mu_phi_0 / v_phi_0;
      log_p1 = log(theta(d)) + 1/2 * log(v_phi_1) + 1/2 * mu_phi_1 * mu_phi_1 / v_phi_1;
      
      // subtract the larger of the two log_p0 and log_p1 from both
      if(log_p0 < log_p1){
        log_p0 -= log_p1;
        log_p1 -= log_p1;
      } else{
        log_p1 -= log_p0;
        log_p0 -= log_p0;
      }
      p1 = exp(log_p1)/(exp(log_p0) + exp(log_p1));
      p0 = exp(log_p0)/(exp(log_p0) + exp(log_p1));
      
      // now we are ready to draw gamma:
      if(gen.uniform() < p1){
        Phi(k,d) = mu_phi_1 + sqrt(v_phi_1)*gen.normal();
        gamma_sum++;
      }
      else Phi(k,d) = 0.0;
    } // closes loop over tasks

    theta(d) = gen.beta(pi.a_theta + gamma_sum, pi.b_theta + di.q - gamma_sum);
  }
}

void update_sigma_uni(double &sigma, sigma_prior_info &sigma_pi, data_info &di, RNG &gen)
{
  double s = 0.0; 
  int n = 0;
  for(size_t i = 0; i < di.n; i++){
    if(di.delta[i] == 1){
      n++;
      s+= di.r_f[i] * di.r_f[i] * di.weight;
    }
  }
  sigma = sqrt((sigma_pi.nu * sigma_pi.lambda + s)/gen.chi_square(sigma_pi.nu + ( (double) n) * di.weight));
}

void update_sigma(std::vector<double> &sigma, std::vector<sigma_prior_info> &sigma_pi, data_info &di, RNG &gen){
  double s = 0.0;
  int n  = 0; // counts the number of observations per task
  for(size_t k = 0; k < di.q; k++){
    s = 0.0;
    n = 0;
    for(size_t i = 0; i < di.n; i++){
      if(di.delta[k + i*di.q] == 1){
        n++;
        s += di.r_f[k + i*di.q] * di.r_f[k + i*di.q] * di.weight;
      }
    } // closes loop over observations
    if(s != s) Rcpp::stop("[update_sigma]: s is nan");
    sigma[k] = sqrt( (sigma_pi[k].nu * sigma_pi[k].lambda + s)/gen.chi_square(sigma_pi[k].nu + ( (double) n) * di.weight));
    if(sigma[k] != sigma[k]){
      Rcpp::Rcout << "[update_sigma]: k = " << k << " s = " << s << endl;
      Rcpp::Rcout << "[update_sigma]: posterior hyperparameters: " << sigma_pi[k].nu * sigma_pi[k].lambda + s << " " << sigma_pi[k].nu + ( (double) n) * di.weight << endl;
      Rcpp::stop("[update_sigma]: sigma[k] is nan");
    }
  } // closes loop over tasks
}


void update_sigma(const arma::mat &Phi, arma::vec &sigma, dinfo_slfm &di, pinfo_slfm &pi, RNG &gen)
{
  double s = 0.0;
  int n = 0;
  for(size_t k = 0; k < di.q; k++){
    n = 0;
    s = 0.0;
    for(size_t i = 0; i < di.n; i++){
      if(di.delta[k + i*di.q] == 1){
        n++;
        s += (di.y[k + i*di.q] - di.af[k + i*di.q]) * (di.y[k + i*di.q] - di.af[k + i*di.q]);
      }
    }
    sigma(k) = sqrt((pi.nu * pi.lambda[k] + s)/gen.chi_square(pi.nu + n));
  } //closes loop over the tasks
}
