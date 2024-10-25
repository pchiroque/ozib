% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/Hypers.R
\name{Hypers}
\alias{Hypers}
\title{Create hyperparameter object for betabart}
\usage{
Hypers(
  X,
  Y,
  W,
  delta1,
  delta0,
  group = NULL,
  alpha = 1,
  beta = 2,
  gamma = 0.95,
  num_tree = 50,
  k = 2,
  k_theta = 2,
  alpha_scale = NULL,
  alpha_shape_1 = 0.5,
  alpha_shape_2 = 1,
  sigma_theta = NULL,
  theta_0 = NULL,
  theta_1 = NULL,
  shape = 1
)
}
\arguments{
\item{X}{NxP matrix of training data covariates.}

\item{Y}{Nx1 vector of training data response.}

\item{group}{For each column of X, gives the associated group}

\item{alpha}{Positive constant controlling the sparsity level}

\item{beta}{Parameter penalizing tree depth in the branching process prior}

\item{gamma}{Parameter penalizing new nodes in the branching process prior}

\item{num_tree}{Number of trees in the ensemble}

\item{k}{Related to the signal-to-noise ratio, sigma_mu = 0.5 / (sqrt(num_tree) * k). BART defaults to k = 2.}

\item{alpha_scale}{Scale of the prior for alpha; if not provided, defaults to P}

\item{alpha_shape_1}{Shape parameter for prior on alpha; if not provided, defaults to 0.5}

\item{alpha_shape_2}{Shape parameter for prior on alpha; if not provided, defaults to 1.0}

\item{shape}{Shape parameter for gating probabilities}

\item{sigma_hat}{A prior guess at the conditional variance of Y. If not provided, this is estimated empirically by linear regression.}

\item{width}{Bandwidth of gating probabilities}

\item{num_tree_prob}{Parameter for geometric prior on number of tree}
}
\value{
Returns a list containing the function arguments.
}
\description{
Creates a list which holds all the hyperparameters for use with the betabart
command.
}
