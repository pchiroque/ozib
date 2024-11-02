#' @export

betabart.hurdle.predict <- function(fit){
  yb.hat <- as.data.frame(fit$fit$lambda_hat)
  y1.hat <- as.data.frame(fit$fit$theta1_hat)
  y0.hat <- as.data.frame(fit$fit$theta0_hat)
  k.alpha <- fit$fit$shape
  length.y <- dim(yb.hat)[2]
  
  # one prediction using probability of each case to be one
  y1.predict <- map_df(y1.hat,pnorm)%>%
    map_df(function(x) sapply(x,function(p)rbinom(1,1,p)))

  # The expected value of the yb
  yb.predict <- map_df(yb.hat,function(a)sapply(a,function(a)rbeta(1,k.alpha*a,k.alpha)))

  # probability of each case to be zero
  y0.predict <- map_df(y0.hat,pnorm)%>%
    map_df(function(x) sapply(x,function(p)rbinom(1,1,p)))
  
  y.predict <- do.call(cbind,lapply(1:length.y,function(i)ifelse(y1.predict[[i]]==0,(y1.predict[[i]]==0)*(ifelse(y0.predict[[i]]==0,yb.predict[[i]],0)),1)))

  y.predict
}
