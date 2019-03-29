library(Rcpp)
library(RcppArmadillo)

source("scripts/makeCutpoints.R")

load("data/toy_example2.RData")
cutpoints <- makeCutpoints(X_train, gridlen = 10000)


source
sourceCpp("src/slfm_bart.cpp")
sourceCpp("src/sep_bartFit.cpp")

sep_test <- sep_bartFit(Y, X_train, X_test, cutpoints, burn = 500, nd = 1000)

slfm_test_D5_m200 <- slfm_bartFit(Y, X_train, X_test, cutpoints, burn = 500, nd = 1000, D = 5, m = 200)
slfm_test_D10_m200 <- slfm_bartFit(Y, X_train, X_test, cutpoints, burn = 500, nd = 1000, D = 10, m = 200)
slfm_test_D20_m200 <- slfm_bartFit(Y, X_train, X_test, cutpoints, burn = 500, nd = 1000, D = 20, m = 200)
slfm_test_D50_m200 <- slfm_bartFit(Y, X_train, X_test, cutpoints, burn = 500, nd = 1000, D = 50, m = 200)

slfm_test_D5_m100 <- slfm_bartFit(Y, X_train, X_test, cutpoints, burn = 500, nd = 1000, D = 5, m = 100)
slfm_test_D10_m100 <- slfm_bartFit(Y, X_train, X_test, cutpoints, burn = 500, nd = 1000, D = 5, m = 100)
slfm_test_D20_m100 <- slfm_bartFit(Y, X_train, X_test, cutpoints, burn = 500, nd = 1000, D = 5, m = 100)
slfm_test_D50_m100 <- slfm_bartFit(Y, X_train, X_test, cutpoints, burn = 500, nd = 1000, D = 5, m = 100)

slfm_test_D5_m50 <- slfm_bartFit(Y, X_train, X_test, cutpoints, burn = 500, nd = 1000, D = 5, m = 50)
slfm_test_D10_m50 <- slfm_bartFit(Y, X_train, X_test, cutpoints, burn = 500, nd = 1000, D = 5, m = 50)
slfm_test_D20_m50 <- slfm_bartFit(Y, X_train, X_test, cutpoints, burn = 500, nd = 1000, D = 5, m = 50)
slfm_test_D50_m50 <- slfm_bartFit(Y, X_train, X_test, cutpoints, burn = 500, nd = 1000, D = 5, m = 50)


slfm_test_D5_m10 <- slfm_bartFit(Y, X_train, X_test, cutpoints, burn = 500, nd = 1000, D = 5, m = 50)
slfm_test_D10_m10 <- slfm_bartFit(Y, X_train, X_test, cutpoints, burn = 500, nd = 1000, D = 5, m = 50)
slfm_test_D20_m10 <- slfm_bartFit(Y, X_train, X_test, cutpoints, burn = 500, nd = 1000, D = 5, m = 50)
slfm_test_D50_m10 <- slfm_bartFit(Y, X_train, X_test, cutpoints, burn = 500, nd = 1000, D = 5, m = 50)

slfm_test_D5_m5 <- slfm_bartFit(Y, X_train, X_test, cutpoints, burn = 500, nd = 1000, D = 5, m = 50)
slfm_test_D10_m5 <- slfm_bartFit(Y, X_train, X_test, cutpoints, burn = 500, nd = 1000, D = 5, m = 50)
slfm_test_D20_m5 <- slfm_bartFit(Y, X_train, X_test, cutpoints, burn = 500, nd = 1000, D = 5, m = 50)
slfm_test_D50_m5 <- slfm_bartFit(Y, X_train, X_test, cutpoints, burn = 500, nd = 1000, D = 5, m = 50)


slfm_test_D5_m1 <- slfm_bartFit(Y, X_train, X_test, cutpoints, burn = 500, nd = 1000, D = 5, m = 50)
slfm_test_D10_m1 <- slfm_bartFit(Y, X_train, X_test, cutpoints, burn = 500, nd = 1000, D = 5, m = 50)
slfm_test_D20_m1 <- slfm_bartFit(Y, X_train, X_test, cutpoints, burn = 500, nd = 1000, D = 5, m = 50)
slfm_test_D50_m1 <- slfm_bartFit(Y, X_train, X_test, cutpoints, burn = 500, nd = 1000, D = 5, m = 50)

# Compare RMSEs



sqrt( mean( (f2_test - rowMeans(sep_test$test_samples[,2,]))^2))
sqrt( mean( (f2_test - rowMeans(slfm_test_D5_m1$f_test_samples[,2,]))^2))




plot(X_test[,1], rowMeans(sep_test$test_samples[,1,]))
points(X_test[,1], f1_test, pch = 4)


plot(X_test[,1], rowMeans(slfm_test_D5_m200$f_test_samples[,5,]))
points(X_test[,1], f5_test, pch = 4)
