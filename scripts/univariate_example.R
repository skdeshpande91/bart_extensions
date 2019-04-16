# Example of univariate BART
library(MASS)
library(Rcpp)
library(RcppArmadillo)

source("scripts/makeCutpoints.R")
sourceCpp("src/uni_bartFit.cpp")

#################################
# Generate some data from a GP whose kernel 
#is the product of a squared exponential kernel and a periodic kernel
#################################

set.seed(41319)
n <- 1000
sigma <- 0.6  # residual variance

X_train <- matrix(sort(runif(n)), ncol = 1)
X_test <- matrix(seq(0, 1, length = 101), ncol = 1)

cutpoints <- makeCutpoints(X_train)

x_all <- c(X_train, X_test)

k_all <- 1*exp(-1/(2 * 0.1 * 0.1) * outer(x_all, x_all, FUN = "-") * outer(x_all, x_all, FUN = "-")) * exp(-2/(0.1*0.1) * sin(pi*outer(x_all, x_all, FUN = "-")/2) * sin(pi*outer(x_all, x_all, FUN = "-")/2))
f_all <- mvrnorm(n = 1, mu = rep(0, times = length(x_all)), Sigma = k_all)

f_train <- f_all[1:n]
f_test <- f_all[(n+1):length(x_all)]

Y <- f_train + sigma * rnorm(n, 0, 1)

f_range <- c(-1,1)*max(abs(f_all))

sort_x_all <- sort(x_all, index.return = TRUE)
plot(sort_x_all$x, f_all[sort_x_all$ix], type = "l", ylim = f_range, col = 'red', xlab = "X", ylab = "f(X)")
points(X_train, Y, pch = 16, cex = 0.4, col = 'black')

fit <- uni_bartFit(Y, X_train, X_test, cutpoints)
points(X_test, rowMeans(fit$test_samples), pch = 4, cex = 0.4, col = 'blue')
