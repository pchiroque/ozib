## Installation
Install those packages via `remotes::install_github("pchiroque/ozib")` and `remotes::install_github("pchiroque/betabart")`.

## Usage 

The response `y` must be a vector with $y \in[0,1]$.

The covariates must be a ´data.frame` and have the same numbers of the rows as the response without intercept.

### Example 

```R

set.seed(4)

rm(list = ls())

remotes::install_github("pchiroque/ozib")
remotes::install_github("pchiroque/betabart")

library(ozib)
library(betabart)

hypers <- ozib::Hypers(X = betabart_data$X, Y = betabart_data$Y,
                       W = betabart_data$X,
                       delta1 = betabart_data$Ydivided$y1,
                       delta0 = betabart_data$Ydivided$y0)
opts   <- ozib::Opts()
opts$approximate_density <- TRUE

time <- Sys.time()
fit <- ozib::betabart(X = betabart_data$X, Y = betabart_data$Y,
                      delta1 = betabart_data$Ydivided$y1,
                      delta0 = betabart_data$Ydivided$y0,
                      yb = betabart_data$Ydivided$yb,
                      X_test = betabart_data$X,
                      hypers_ = hypers, opts_ = opts)
print(duration <- Sys.time() - time)

fit <- list(y = betabart_data$Y, x = betabart_data$X, fit = fit)

y.predict <- ozib::betabart.hurdle.predict(fit)

fig.bart <- ozib::betabart.hurdle.validation(y.predict, fit$y)

fig.bart

```




