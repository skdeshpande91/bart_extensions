# Generate data
library(MASS)
library(Rcpp)
library(RcppArmadillo)
source("scripts/makeCutpoints.R")
sourceCpp("src/rbart.cpp")


# Specify training points
n_pred <- 1000

p <- 1
D <- 5
q <- 5
X_pred <- matrix(seq(from = -1, to = 1, length = n_pred), nrow = n_pred, ncol = 1)

cutpoints <- makeCutpoints(X_pred)
# Need to call make-cutpoints


test <- rbart(X_pred, cutpoints, n_samples = D, m = 200, sigma_mu = 1/(2 * 3 * sqrt(200)))

plot(X_pred, test[1,], ylim = range(test), type = "l")
lines(X_pred, test[2,], col = 'red')
lines(X_pred, test[3,], col = 'green')
lines(X_pred, test[4,], col = 'blue')
lines(X_pred, test[5,], col = 'purple')


Phi <- matrix(rnorm(D*q, 0, 1), nrow = q, ncol = D)
f <- t(Phi %*% test)


plot(X_pred, f[,1], type = "l", ylim = range(f))
lines(X_pred, f[,2], col = 'red')
lines(X_pred, f[,3], col = 'blue')
lines(X_pred, f[,4], col = 'green')
lines(X_pred, f[,5], col = 'purple')
for(k in 1:q){
  
  
  
  f[,k] <- Phi[k,] * test[]
}




