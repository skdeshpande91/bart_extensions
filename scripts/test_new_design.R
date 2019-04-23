library(Rcpp)
library(RcppArmadillo)
library(MASS)
source("scripts/makeCutpoints.R")

sourceCpp("src/univariate_BART.cpp")
sourceCpp("src/sep_BART.cpp")
sourceCpp("src/slfm_BART.cpp")


n <- 1000
sigma_1 <- 0.75
sigma_2 <- 0.5

set.seed(31119)
X_train <- cbind("X1" = runif(n), "X2" = 1*(runif(n) > 0.5))

X_test <- cbind("X1" = c(seq(0, 1, by = 0.01), seq(0,1,by = 0.01)), "X2" = c(rep(0, times = 101), rep(1, times = 101)))

cutpoints <- makeCutpoints(X_train)

x1_test <- X_test[X_test[,2] == 0,1]

f1_train <- 5 * X_train[,1] * X_train[,1]
f1_test <- 5 * x1_test * x1_test
f1_all <- c(f1_train, f1_test)


f2_train <- 3 * X_train[,1] + (2 - 5 * (X_train[,2] > 0.5)) * sin(pi * X_train[,1]) - 2*(X_train[,2] > 0.5)
f2_test_0 <- 3 * x1_test + (2 - 5*0) * sin(pi * x1_test) - 2*0
f2_test_1 <- 3 * x1_test + (2 - 5*1) * sin(pi * x1_test) - 2*1
f2_all <- c(f2_train, f2_test_0, f2_test_1)


Y1 <- f1_train + sigma_1 * rnorm(n, 0, 1)
Y2 <- f2_train + sigma_2 * rnorm(n,0,1)

Y <- matrix(cbind(Y1,Y2), nrow = n, ncol = 2)

test1 <- univariate_BART(Y[,1], X_train, X_test, cutpoints, verbose = TRUE)
test2 <- univariate_BART(Y[,2], X_train, X_test, cutpoints, verbose = TRUE)

sep_test <- sep_BART(Y, X_train, X_test, cutpoints, verbose = TRUE,  nd = 5, burn = 100)


sourceCpp("src/slfm_bartFit.cpp")

slfm_test <- slfm_BART(Y, X_train, X_test, cutpoints, D = 10, m = 25)

old_slfm_test <- slfm_bartFit(Y, X_train, X_test, cutpoints, D = 10, m = 25)


Y_miss <- Y
Y_miss[sample(1:n,10), 1] <- NA
Y_miss[sample(1:n, 20), 2] <- NA

test1_missing <- univariate_BART(Y_miss[,1], X_train, X_test, cutpoints, verbose = TRUE)
test2_missing <- univariate_BART(Y_miss[,2], X_train, X_test, cutpoints, verbose = TRUE)

sep_test_missing <- sep_BART(Y_miss, X_train, X_test, cutpoints, verbose = TRUE)

sep_fit <- sep_bartFit(Y, X_train, X_test, cutpoints, verbose = TRUE,)

# check to see our new code univariate_BART gives similar predictive performance results

sqrt( (mean( (f1_test - rowMeans(test1$f_test_samples))^2)))
sqrt( (mean( (f1_test - rowMeans(sep_test$f_test_samples[,1,]))^2)))
sqrt( (mean( (f1_test - rowMeans(test1_missing$f_test_samples))^2)))
sqrt( (mean( (f1_test - rowMeans(sep_test_missing$f_test_samples[,1,]))^2)))

sqrt( (mean((c(f2_test_0, f2_test_1) - rowMeans(test2$f_test_samples))^2)))
sqrt( (mean((c(f2_test_0, f2_test_1) - rowMeans(sep_test$f_test_samples[,2,]))^2)))
sqrt( (mean((c(f2_test_0, f2_test_1) - rowMeans(test2_missing$f_test_samples))^2)))
sqrt( (mean((c(f2_test_0, f2_test_1) - rowMeans(sep_test_missing$f_test_samples[,2,]))^2)))


sqrt( (mean((c(f2_test_0, f2_test_1) - rowMeans(sep_fit$f_test_samples[,2,]))^2)))

