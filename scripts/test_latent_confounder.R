library(Rcpp)
library(RcppArmadillo)

sourceCpp("src/latent_confounder_BART.cpp")
sourceCpp("src/univariate_BART.cpp")

source("scripts/makeCutpoints.R")

n <- 1000
p_obs <- 4
sigma <- 0.1

X <- matrix(rnorm(n*p), n, p)
U <- rbinom(n, 1, 0.5)
e <- sigma * rnorm(n, 0, 1)

f0 <- as.vector(0.15 * X[,1]^2 + 0.15 * X[,2]^2 + 0.15 * X[,3] * X[,4])
f1 <- f0 + as.vector(0.85 * U * X[,3] * X[,4])

Y0 <- f0 + e
Y1 <- f1 + e

cutpoints_x <- makeCutpoints(X)
cutpoints_xu <- makeCutpoints(cbind(X,U))

ufit0_x <- univariate_BART(Y0, X, X, cutpoints_x, verbose = TRUE)
ufit1_x <- univariate_BART(Y1, X, X, cutpoints_x, verbose = TRUE)
ufit0_ux <- univariate_BART(Y0, cbind(X,U), cbind(X,U), cutpoints_xu, verbose = TRUE)
ufit1_ux <- univariate_BART(Y1, cbind(X,U), cbind(X,U), cutpoints_xu, verbose = TRUE)


test0 <- lat_conf_BART(Y0, X, verbose = TRUE)
test1 <- lat_conf_BART(Y1, X, verbose = TRUE)


sqrt(mean( (f0 - rowMeans(ufit0_x$f_train_samples))^2))
sqrt(mean( (f0 - rowMeans(ufit0_ux$f_train_samples))^2))
sqrt(mean( (f0 - rowMeans(test0$f_train_samples))^2))


sqrt(mean( (f1 - rowMeans(ufit1_x$f_train_samples))^2))
sqrt(mean( (f1 - rowMeans(ufit1_ux$f_train_samples))^2))
sqrt(mean( (f1 - rowMeans(test1$f_train_samples))^2)) # just about as bad as not including U...

