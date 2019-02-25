library(Rcpp)
library(RcppArmadillo)

source("scripts/makeCutpoints.R")
n <- 10
p <- 2
q <- 3
X <- matrix(runif(n*p, -1, 1), n, p)
X_pred <- X
Y <- matrix(rnorm(n*q), n,q)

cutpoints <- makeCutpoints(X)

sourceCpp("src/slfm_bart.cpp")
sourceCpp("src/test_backfitting.cpp")


Y[1,1] <- NA
Y[8,2] <- NA
Y[c(1,3,4), 3] <- NA

D <- 5
Phi <- matrix(0, nrow = q,ncol = D)
Phi[1,c(1,2,3)] <- 1
Phi[2,c(4,5)] <- 1
Phi[3, ] <- 1

sigma <- rep(1, times = D)


test <- test_backfitting(Y, X, X_pred, cutpoints, Phi, sigma)

# To really test this, we need to implement a version in R ourselves. 
# 


test <- slfm_bartFit(Y, X, X_pred,cutpoints)
