#ifndef RECBART_H
#define RECBART_H

#include <RcppArmadillo.h>
#include "functions.h"
#include "slice.h"

// THINGS TO CHANGE WHEN ADDING NEW FEATURES:
// - The parameters in Hypers should be changed to reflect the current situation
// - By design, none of the changes to tree topology will need to be changed
// - MyData should be changed to reflect changes to the format of the data
//   - My Data includes also the estimated forest predictions (or, leave-one-out predictions)
//     to be used in the algorithm. It might contain additional information; the important
//     point is that it includes the relevant leave-one-out information to be used when
//     computed LogLT
// - In general, LogLT will need to be changed to reflect what is going on in the model
// - Other than these changes, you will also need to write new updates for the parameters.
//   - These type of updates are Node::UpdateMu()
//   - Depending on how predictions are being done, predict may need to be changed
// - Depending on the nature of the backfitting, you will also need to change
//   TreeBackfit, IterateGibbsNoS, and IterateGibbsWithS
// - The change to TreeBackfit is done by changing the functions Backfit and Refit
// - Need to change SuffStats, which is also model specific

struct Hypers;
struct Node;
struct MyData;

struct Hypers {

  // Tree parameters
  double alpha;
  double gamma;
  double beta;
  int num_trees;

  // Gamma parameters
  double a_lambda;
  double b_lambda;
  double shape;
  double lambda_0;

  // Classification parameters
  double theta_0;
  double sigma_theta0;
  double theta_1;
  double sigma_theta1;

  // Alpha hyperparameters
  double alpha_scale;
  double alpha_shape_1;
  double alpha_shape_2;

  // Variance hyperparameters
  /* double sigma_theta_hat; */
  /* double sigma_mu_hat; */

  // splitting prob parameters
  arma::vec s;
  arma::vec logs;

  // Grouping
  int num_groups;
  arma::uvec group;
  std::vector<std::vector<unsigned int> > group_to_vars;

  // Constructor
  Hypers(const arma::mat& X, const arma::uvec& group, Rcpp::List hypers);

  // Functions
  void UpdateAlpha();
  /* void UpdateTau(MyData& mydata); */
  int SampleVar() const;

};

struct MyData {
  arma::mat X;
  arma::vec Y;
  arma::vec Z0;
  arma::vec Z1;
  arma::uvec delta1;
  arma::uvec delta0;
  arma::vec yb;
  arma::vec lambda_hat;
  arma::vec theta0_hat;
  arma::vec theta1_hat;
  
  int nb;
  double sumlog1my;

  MyData(arma::mat& Xx, arma::vec& Yy,
         arma::uvec& delta1x, arma::uvec& delta0x, arma::vec& ybeta, double lambda_0,
         double theta_0, double theta_1)
  : X(Xx), Y(Yy), delta1(delta1x), delta0(delta0x), yb(ybeta) {

    lambda_hat = lambda_0 * arma::ones<arma::vec>(Xx.n_rows);
    theta0_hat = theta_0 + arma::zeros<arma::vec>(Xx.n_rows);
    theta1_hat = theta_1 + arma::zeros<arma::vec>(Xx.n_rows);
    Z0 = arma::zeros<arma::vec>(delta0.n_elem);
    Z1 = arma::zeros<arma::vec>(delta1.n_elem);

    for(int i = 0; i < delta1.n_elem; i++) {
      if (Rcpp::NumericVector::is_na(delta1(i))){
        Z1(i) = 0;
        continue;
      }
      
      if(delta1(i) == 0) {
        Z1(i) = randnt(theta1_hat(i), 1.0, R_NegInf, 0.0);
      } else {
        Z1(i) = randnt(theta1_hat(i), 1.0, 0.0, R_PosInf);
      }
    }
    for(int i = 0; i < delta0.n_elem; i++) {
      if (Rcpp::NumericVector::is_na(delta0(i))){
        Z0(i) = 0;
        continue;
      }
      
      if(delta0(i) == 0) {
        Z0(i) = randnt(theta0_hat(i), 1.0, R_NegInf, 0.0);
      } else {
        Z0(i) = randnt(theta0_hat(i), 1.0, 0.0, R_PosInf);
      }
    }
    
    nb = 0;
    sumlog1my = 0.;
    for (double yybb : yb) {
      if (!Rcpp::NumericVector::is_na(yybb)) {
        nb++;
        sumlog1my += log(1 - yybb);
      }
    }
  }
};

struct SuffStats {
  double sum_v_Y;
  double sum_log_Y;
  double sum_log_v;
  double n;
  double sum_Z1;
  double sum_Z1_sq;
  double n_Z1;
  double sum_Z0;
  double sum_Z0_sq;
  double n_Z0;

SuffStats() : sum_v_Y(0.0), sum_log_Y(0.0), sum_log_v(0.0), n(0.0),
sum_Z1(0.0), sum_Z1_sq(0.0), n_Z1(0.0),
sum_Z0(0.0), sum_Z0_sq(0.0), n_Z0(0.0){;}

};

struct Node {
  bool is_leaf;
  bool is_root;
  Node* left;
  Node* right;
  Node* parent;

  // Branch parameters
  int var;
  double val;
  double lower;
  double upper;
  int depth;

  // Leaf parameters
  double lambda;
  double theta0;
  double theta1;

  SuffStats ss;

  // Hyperparameters
  Hypers* hypers;

  void Root();
  void GetLimits();
  void BirthLeaves();
  bool is_left();
  void DeleteLeaves();
  void UpdateParams(MyData& data);
  double LogLT(const MyData& data);
  void UpdateSuffStat(const MyData& data);
  void ResetSuffStat();
  void AddSuffStat(const MyData& data, int i);
  void AddSuffStatZ(const MyData& data, int i);

  Node(Hypers* hypers_);
  Node(Node* parent);
  ~Node();
};

struct Opts {
  int num_burn;
  int num_thin;
  int num_save;
  int num_print;

  bool update_s;
  bool update_alpha;
  
  bool approximate_density;

  Opts() : update_s(true), update_alpha(true) {

    num_burn = 1;
    num_thin = 1;
    num_save = 1;
    num_print = 100;

  }

  Opts(int nburn, int nthin, int nsave,
       int nprint, bool updates, bool updatealpha) :
  num_burn(nburn), num_thin(nthin), num_save(nsave), num_print(nprint),
  update_s(updates), update_alpha(updatealpha), approximate_density(false) {;}

  Opts(Rcpp::List opts) {
    num_burn = opts["num_burn"];
    num_thin = opts["num_thin"];
    num_save = opts["num_save"];
    num_print = opts["num_print"];
    update_s = opts["update_s"];
    update_alpha = opts["update_alpha"];
    approximate_density = opts["approximate_density"];
  }
};

// Node functions
double growth_prior(Node* node);
int depth(Node* node);
void leaves(Node* x, std::vector<Node*>& leafs);
std::vector<Node*> leaves(Node* x);
std::vector<Node*> not_grand_branches(Node* tree);
void not_grand_branches(std::vector<Node*>& ngb, Node* node);
Node* rand(std::vector<Node*> ngb);
arma::uvec get_var_counts(std::vector<Node*>& forest, const Hypers& hypers);
void get_var_counts(arma::uvec& counts, Node* node, const Hypers& hypers);
double forest_loglik(std::vector<Node*>& forest);
double tree_loglik(Node* node);
std::vector<Node*> init_forest(Hypers& hypers);

// Tree MCMC
double probability_node_birth(Node* tree);
Node* birth_node(Node* tree, double* leaf_node_probability);
Node* death_node(Node* tree, double* p_not_grand);
void birth_death(Node* tree, const MyData& data);
void node_birth(Node* tree, const MyData& data);
void node_death(Node* tree, const MyData& data);
void change_decision_rule(Node* tree, const MyData& data);

// Other MCMC
void UpdateS(std::vector<Node*>& forest);
void UpdateZ(MyData& data);
/* void UpdateSigmaParam(std::vector<Node*>& forest); */
/* arma::mat get_params(std::vector<Node*>& forest); */
/* void get_params(Node* n, std::vector<double>& mu, */
/*                 std::vector<double>& tau, std::vector<double>& theta); */
void UpdateShape(Hypers* hypers, MyData& data, bool approximate);


/* arma::vec loglik_data(const arma::vec& Y, const arma::vec& rho, const Hypers& hypers); */
void IterateGibbsNoS(std::vector<Node*>& forest,
                     MyData& data,
                     const Opts& opts);
void IterateGibbsWithS(std::vector<Node*>& forest,
                       MyData& data,
                       const Opts& opts);
void do_burn_in(std::vector<Node*> forest, MyData& data, const Opts& opts);
void TreeBackfit(std::vector<Node*>& forest,
                 MyData& mydata,
                 const Opts& opts);
void BackFit(Node* tree, MyData& data);
void Refit(Node* tree, MyData& data);

// Predictions
arma::vec predict_reg(Node* tree, MyData& data);
arma::vec predict_reg(Node* tree, arma::mat& X);
double predict_reg(Node* n, arma::rowvec& x);
arma::vec predict_reg(std::vector<Node*> forest, arma::mat& X);
arma::vec predict_theta1(Node* tree, MyData& data);
arma::vec predict_theta1(Node* tree, arma::mat& W);
arma::vec predict_theta0(Node* tree, MyData& data);
arma::vec predict_theta0(Node* tree, arma::mat& W);
double predict_theta1(Node* tree, arma::rowvec& x);
double predict_theta0(Node* tree, arma::rowvec& x);
arma::vec predict_theta1(std::vector<Node*> forest, arma::mat& W);
arma::vec predict_theta0(std::vector<Node*> forest, arma::mat& W);

#endif
