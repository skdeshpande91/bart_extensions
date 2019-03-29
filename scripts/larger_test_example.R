# Generate slightly more complicated SLFM
library(MASS)
source("scripts/makeCutpoints.R")

n <- 1000
p <- 5
q <- 10
D <- 20


# Let all of the data belong to unit cube [0,1]^d
set.seed(322)
X_train <- matrix(runif(n*p, 0, 1), nrow = n, ncol = p)
X_test <- matrix(runif(100*p,,1), nrow = n, ncol = p)

X_all <- rbind(X_train, X_test)

sigma <- runif(n = q, 0.25,1) # residual variances



# Just generate from a bunch of GP's

# allow us to use SE kernels and periodic kernels
k_per <- function(x, l){
  return(exp(-2/(l*l) * sin(pi*outer(x, x, FUN= "-")/2) * sin(pi*outer(x, x, FUN = "-")/2)))
}
k_se <- function(x, l){
  return(exp(-1/(2 * l * l) * outer(x, x, FUN = "-") * outer(x, x, FUN = "-")))
}



for(d in 1:D){
  n_vars <- rbinom(1, p, 0.5)
  while(n_vars == 0) n_vars <- rbinom(1, p, 0.5)
  vars <- sample(1:p)[1:n_vars] # sample the directions
  
  
}
