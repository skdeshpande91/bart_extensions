# Fix sigma and Phi in sep_bart and slfm_bart and see how well we do
library(Rcpp)
library(RcppArmadillo)

source("scripts/makeCutpoints.R")
load("data/toy_example.RData")

cutpoints <- makeCutpoints(X_train, gridlen = 10000)
sigma_orig <- c(sigma_1, sigma_2)


# What if we just fix sigma in slfm_bartFit
sourceCpp("src/slfm_bart.cpp")
sourceCpp("src/slfm_bart_fixed_sigma.cpp")


slfm_test <- slfm_bartFit(Y, X_train, X_test, cutpoints, nd = 1000, burn = 500, D = 4, m = 200)

slfm_test2 <- slfm_bartFit(Y, X_train, X_test, cutpoints, nd = 1000, burn = 500, D = 10, m = 200)
slfm_test3 <- slfm_bartFit(Y, X_train, X_test, cutpoints, nd = 1000, burn = 500, D = 2, m = 200)
slfm_test4 <- slfm_bartFit(Y, X_train, X_test, cutpoints, nd = 1000, burn = 500, D = 25, m = 100)



slfm_test_sigma <- slfm_bartFit_fixed_sigma(Y, X_train, X_test, cutpoints, sigma_init = sigma_orig, nd = 1000, 
                                            burn = 500, D = 4, m = 200)


plot(1, type = "n", xlab = expression(X[1]), ylab = "f", main = "Posterior Mean from slfmBART fits", xlim = c(0,1), ylim = f_range)

points(X_test[,1], rowMeans(slfm_test4$f_test_samples[,1,]), pch = 16, cex = 0.5, col = 'red')
points(X_test[,1], rowMeans(slfm_test4$f_test_samples[,2,]), pch = 4, cex = 0.5, col = 'blue')

lines(X_test[1:101,1], f1_test_0, col = 'red', lty = 1)
lines(X_test[1:101,1], f1_test_1, col = 'red', lty= 2)
lines(X_test[1:101,1], f2_test_0, col = 'blue', lty = 1)
lines(X_test[1:101,1], f2_test_1, col = 'blue', lty = 2)


plot(1, type = "n", xlab = expression(X[1]), ylab = "f", main = "Posterior Mean from slfmBART fits: fixed sigma", xlim = c(0,1), ylim = f_range)

points(X_test[,1], rowMeans(slfm_test_sigma$f_test_samples[,1,]), pch = 16, cex = 0.5, col = 'red')
points(X_test[,1], rowMeans(slfm_test_sigma$f_test_samples[,2,]), pch = 4, cex = 0.5, col = 'blue')

lines(X_test[1:101,1], f1_test_0, col = 'red', lty = 1)
lines(X_test[1:101,1], f1_test_1, col = 'red', lty= 2)
lines(X_test[1:101,1], f2_test_0, col = 'blue', lty = 1)
lines(X_test[1:101,1], f2_test_1, col = 'blue', lty = 2)

# RMSE
sqrt( mean( (c(f1_test_0, f1_test_1) - rowMeans(slfm_test$f_test_samples[,1,]))^2))
sqrt( mean( (c(f2_test_0, f2_test_1) - rowMeans(slfm_test$f_test_samples[,2,]))^2))

sqrt( mean( (c(f1_test_0, f1_test_1) - rowMeans(slfm_test2$f_test_samples[,1,]))^2))
sqrt( mean( (c(f2_test_0, f2_test_1) - rowMeans(slfm_test2$f_test_samples[,2,]))^2))

sqrt( mean( (c(f1_test_0, f1_test_1) - rowMeans(slfm_test3$f_test_samples[,1,]))^2))
sqrt( mean( (c(f2_test_0, f2_test_1) - rowMeans(slfm_test3$f_test_samples[,2,]))^2))

sqrt( mean( (c(f1_test_0, f1_test_1) - rowMeans(slfm_test4$f_test_samples[,1,]))^2))
sqrt( mean( (c(f2_test_0, f2_test_1) - rowMeans(slfm_test4$f_test_samples[,2,]))^2))

sqrt( mean( (c(f1_test_0, f1_test_1) - rowMeans(slfm_test_sigma$f_test_samples[,1,]))^2))
sqrt( mean( (c(f2_test_0, f2_test_1) - rowMeans(slfm_test_sigma$f_test_samples[,2,]))^2))




# Check that fixing sigma doesn't break sep_bartFit
sqrt( mean( (c(f1_test_0, f1_test_1) - rowMeans(sep_fit$test_samples[,1,]))^2))
sqrt( mean( (c(f2_test_0, f2_test_1) - rowMeans(sep_fit$test_samples[,2,]))^2))

sqrt( mean( (c(f1_test_0, f1_test_1) - rowMeans(sep_fit_sigma$test_samples[,1,]))^2))
sqrt( mean( (c(f2_test_0, f2_test_1) - rowMeans(sep_fit_sigma$test_samples[,2,]))^2))

sourceCpp("src/sep_bartFit.cpp")
sourceCpp("src/sep_bartFit_fixed_sigma.cpp")


sep_fit <- sep_bartFit(Y, X_train, X_test, cutpoints)

sep_fit_sigma <- sep_bartFit_fixed_sigma(Y, X_train, X_test, sigma_init = sigma_orig, cutpoints)


# RMSE's
# fixed sigma
sqrt( mean((Y[,1] - rowMeans(sep_fit_sigma$train_samples[,1,]))^2))
sqrt( mean((Y[,1] - rowMeans(sep_fit$train_samples[,1,]))^2))

sqrt( mean( (Y[,2] - rowMeans(sep_fit_sigma$train_samples[,2,]))^2))
sqrt( mean( (Y[,2] - rowMeans(sep_fit$train_samples[,2,]))^2))

sqrt(mean( (f1_train - rowMeans(sep_fit_sigma$train_samples[,1,]))^2))
sqrt(mean( (f1_train - rowMeans(sep_fit$train_samples[,1,]))^2))

sqrt(mean( (f2_train - rowMeans(sep_fit_sigma$train_samples[,2,]))^2))
sqrt(mean( (f2_train - rowMeans(sep_fit$train_samples[,2,]))^2))

sqrt( mean( (c(f1_test_0, f1_test_1) - rowMeans(sep_fit_sigma$test_samples[,1,]))^2))
sqrt( mean( (c(f1_test_0, f1_test_1) - rowMeans(sep_fit$test_samples[,1,]))^2))

sqrt( mean( (c(f2_test_0, f2_test_1) - rowMeans(sep_fit_sigma$test_samples[,2,]))^2))
sqrt( mean( (c(f2_test_0, f2_test_1) - rowMeans(sep_fit$test_samples[,2,]))^2))


##############

par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0))
plot(1, type = "n", xlab = expression(X[1]), ylab = "f", main = "Posterior Mean from separate BART fits: fixed sigma", xlim = c(0,1), ylim = f_range)

points(X_test[,1], rowMeans(sep_fit_sigma$test_samples[,1,]), pch = 16, cex = 0.5, col = 'red')
points(X_test[,1], rowMeans(sep_fit_sigma$test_samples[,2,]), pch = 4, cex = 0.5, col = 'blue')

lines(X_test[1:101,1], f1_test_0, col = 'red', lty = 1, lwd = 2)
lines(X_test[102:202,1], f1_test_1, col = 'red', lty = 2, lwd = 2)

lines(X_test[1:101,1], f2_test_0, col = 'blue', lty = 1, lwd = 2)
lines(X_test[102:202,1], f2_test_1, col = 'blue', lty = 2, lwd = 2)
legend("bottomleft", legend = c(expression(x[2]==0), expression(x[2]==1)), lty = c(1,2), bty = "n", col = c('red', 'blue'))
legend("topleft", legend = c(expression(f[1]), expression(f[2])), pch = c(16, 4), col = c('red', 'blue'),text.col = c('red', 'blue'), bty = "n")
