library(MASS)
library(Rcpp)
library(RcppArmadillo)

source("scripts/makeCutpoints.R")



sourceCpp("src/sep_bartFit.cpp")

load("data/toy_example.RData")


cutpoints <- makeCutpoints(X_train, gridlen = 10000)

sep_fit <- sep_bartFit(Y, X_train, X_test, cutpoints, verbose = TRUE)

Y_missing <- Y
Y_missing[1:10,1] <- NA
sep_fit_missing <- sep_bartFit(Y_missing, X_train, X_test, cutpoints, verbose = TRUE)


# RMSE
sqrt(mean( (c(f1_test_0, f1_test_1) - rowMeans(sep_fit$f_test_samples[,1,]))^2))
sqrt(mean( (c(f1_test_0, f1_test_1) - rowMeans(sep_fit_missing$f_test_samples[,1,]))^2))

sqrt(mean( (c(f2_test_0, f2_test_1) - rowMeans(sep_fit$f_test_samples[,2,]))^2))
sqrt(mean( (c(f2_test_0, f2_test_1) - rowMeans(sep_fit_missing$f_test_samples[,2,]))^2))

sqrt( mean( (Y[,1] - rowMeans(sep_fit_missing$f_train_samples[,1,]))^2))

plot(X_train[,1], rowMeans(sep_fit$f_train_samples[,1,]))

# Don't proceed below 28 March 2019

sourceCpp("src/sep_bartFit.cpp")
sourceCpp("src/slfm_bart.cpp")


load("data/toy_example.RData")


cutpoints <- makeCutpoints(X_train, gridlen = 10000)


#####
# First do seperate BART fits

sep_fit <- sep_bartFit(Y, X_train, X_test, cutpoints, verbose = TRUE)

Y_missing <- Y
Y_missing[1:10,1] <- NA
sep_fit_missing <- sep_bartFit(Y_missing, X_train, X_test, cutpoints, verbose = TRUE)

slfm_test_D2_m100 <- slfm_bartFit(Y, X_train, X_test, cutpoints, nd = 1000, burn = 500, D = 2, m = 100)
slfm_test_D2_m200 <- slfm_bartFit(Y, X_train, X_test, cutpoints, nd = 1000, burn = 500, D = 2, m = 200)
slfm_test_D4_m100 <- slfm_bartFit(Y, X_train, X_test, cutpoints, nd = 1000, burn = 500, D = 4, m = 100)
slfm_test_D4_m200 <- slfm_bartFit(Y, X_train, X_test, cutpoints, nd = 1000, burn = 500, D = 4, m = 200)

slfm_test_D10_m100 <- slfm_bartFit(Y, X_train, X_test, cutpoints, nd = 1000, burn = 500, D = 10, m = 100)
slfm_test_D10_m200 <- slfm_bartFit(Y, X_train, X_test, cutpoints, nd = 1000, burn = 500, D = 10, m = 200)

slfm_test_D100_m1 <- slfm_bartFit(Y, X_train, X_test, cutpoints, nd = 1000, burn = 500, D = 100, m = 1)
slfm_test_D200_m1 <- slfm_bartFit(Y, X_train, X_test, cutpoints, nd = 1000, burn = 500, D = 200, m = 1)

slfm_test_D50_m50 <- slfm_bartFit(Y, X_train, X_test, cutpoints, nd = 1000, burn = 500, D = 50, m = 50)
slfm_test_D50_m200 <- slfm_bartFit(Y, X_train, X_test, cutpoints, nd = 1000, burn = 500, D = 50, m = 200)


save(sep_fit, slfm_test_D2_m100, slfm_test_D2_m200, slfm_test_D4_m100, slfm_test_D4_m200,
     slfm_test_D10_m100, slfm_test_D10_m200, slfm_test_D50_m50, slfm_test_D50_m200, file = "slfm_test_mar20.RData")

# RMSE
sqrt(mean( (c(f1_test_0, f1_test_1) - rowMeans(sep_fit$test_samples[,1,]))^2))
sqrt(mean( (c(f2_test_0, f2_test_1) - rowMeans(sep_fit$test_samples[,2,]))^2))

sqrt(mean( (c(f1_test_0, f1_test_1) - rowMeans(slfm_test_D2_m100$f_test_samples[,1,]))^2))
sqrt(mean( (c(f2_test_0, f2_test_1) - rowMeans(slfm_test_D2_m100$f_test_samples[,2,]))^2))

sqrt(mean( (c(f1_test_0, f1_test_1) - rowMeans(slfm_test_D2_m200$f_test_samples[,1,]))^2))
sqrt(mean( (c(f2_test_0, f2_test_1) - rowMeans(slfm_test_D2_m200$f_test_samples[,2,]))^2))

sqrt(mean( (c(f1_test_0, f1_test_1) - rowMeans(slfm_test_D4_m100$f_test_samples[,1,]))^2))
sqrt(mean( (c(f2_test_0, f2_test_1) - rowMeans(slfm_test_D4_m100$f_test_samples[,2,]))^2))

sqrt(mean( (c(f1_test_0, f1_test_1) - rowMeans(slfm_test_D4_m200$f_test_samples[,1,]))^2))
sqrt(mean( (c(f2_test_0, f2_test_1) - rowMeans(slfm_test_D4_m200$f_test_samples[,2,]))^2))

sqrt(mean( (c(f1_test_0, f1_test_1) - rowMeans(slfm_test_D10_m200$f_test_samples[,1,]))^2))
sqrt(mean( (c(f2_test_0, f2_test_1) - rowMeans(slfm_test_D10_m200$f_test_samples[,2,]))^2))

sqrt(mean( (c(f1_test_0, f1_test_1) - rowMeans(slfm_test_D100_m1$f_test_samples[,1,]))^2))
sqrt(mean( (c(f2_test_0, f2_test_1) - rowMeans(slfm_test_D100_m1$f_test_samples[,2,]))^2))

sqrt(mean( (c(f1_test_0, f1_test_1) - rowMeans(slfm_test_D200_m1$f_test_samples[,1,]))^2))
sqrt(mean( (c(f2_test_0, f2_test_1) - rowMeans(slfm_test_D200_m1$f_test_samples[,2,]))^2))

sqrt(mean( (c(f1_test_0, f1_test_1) - rowMeans(slfm_test_D50_m50$f_test_samples[,1,]))^2))
sqrt(mean( (c(f2_test_0, f2_test_1) - rowMeans(slfm_test_D50_m200$f_test_samples[,2,]))^2))



slfm_test_D4_m200 <- slfm_bartFit(Y, X_train, X_test,)



slfm_test <- slfm_bartFit(Y, X_train, X_test, cutpoints, nd = 1000, burn = 500, D = 4, m = 200)

sourceCpp("src/uni_bartFit.cpp")

sourceCpp("src/sep_bartFit.cpp")
sep_fit <- sep_bartFit(Y, X_train, X_test, cutpoints)

### Plot how good the fits are for separate BART fits
png("images/toy_example_sepBART_fits.png", width = 6, height = 6, units = "in", res = 300)
par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0))
plot(1, type = "n", xlab = expression(X[1]), ylab = "f", main = "Posterior Mean from separate BART fits", xlim = c(0,1), ylim = f_range)

points(X_test[,1], rowMeans(sep_fit$f_test_samples[,1,]), pch = 16, cex = 0.5, col = 'red')
points(X_test[,1], rowMeans(sep_fit$f_test_samples[,2,]), pch = 4, cex = 0.5, col = 'blue')

lines(X_test[1:101,1], f1_test_0, col = 'red', lty = 1, lwd = 2)
lines(X_test[102:202,1], f1_test_1, col = 'red', lty = 2, lwd = 2)

lines(X_test[1:101,1], f2_test_0, col = 'blue', lty = 1, lwd = 2)
lines(X_test[102:202,1], f2_test_1, col = 'blue', lty = 2, lwd = 2)
legend("bottomleft", legend = c(expression(x[2]==0), expression(x[2]==1)), lty = c(1,2), bty = "n", col = c('red', 'blue'))
legend("topleft", legend = c(expression(f[1]), expression(f[2])), pch = c(16, 4), col = c('red', 'blue'),text.col = c('red', 'blue'), bty = "n")

dev.off()

# In-sample RMSE
sqrt( mean((Y[,1] - rowMeans(sep_fit$train_samples[,1,]))^2))
sqrt( mean( (Y[,2] - rowMeans(sep_fit$train_samples[,2,]))^2))

sqrt(mean( (f1_train - rowMeans(sep_fit$train_samples[,1,]))^2))
sqrt(mean( (f2_train - rowMeans(sep_fit$train_samples[,2,]))^2))

sqrt( mean( (c(f1_test_0, f1_test_1) - rowMeans(sep_fit$test_samples[,1,]))^2))
sqrt( mean( (c(f2_test_0, f2_test_1) - rowMeans(sep_fit$test_samples[,2,]))^2))

plot(1, type = "n", xlim  = c(0,1), ylim = f_range)
points(X_test[,1], rowMeans(sep_fit$test_samples[,1,]), pch = 3, cex = 0.5)
lines(X_test[1:101,1], f1_test_0, col = 'red', lty = 1)
lines(X_test[102:202,1], f1_test_1, col = 'red', lty = 2)


points(X_test[,1], rowMeans(sep_fit$test_samples[,2,]), pch = 4, cex = 0.5)
lines(X_test[1:101,1], f2_test_0, col = 'blue', lty = 1)
lines(X_test[102:202,1], f2_test_1, col = 'blue', lty = 2)



# let's just learn u1
test1 <- uni_bartFit(u1_train, X_train, X_test, cutpoints)

test2 <- uni_bartFit(u2_train, X_train, X_test, cutpoints)

test_3 <- uni_bartFit(f1_train, X_train, X_test, cutpoints)
test_4 <- uni_bartFit(f2_train, X_train, X_test, cutpoints)

plot(1, type = "n", xlim = c(0,1), ylim = f_range)
points(X_test[,1], rowMeans(test_3$test_samples))
lines(X_test[1:101,1], f1_test_0, col = 'red', lty = 1)
lines(X_test[102:202,1], f1_test_1, col = 'red', lty = 2)

plot(1, type = "n", xlim = c(0,1), ylim = f_range)
points(X_test[,1], rowMeans(test_4$test_samples))
lines(X_test[1:101,1], f2_test_0, col = 'blue', lty = 1)
lines(X_test[102:202,1], f2_test_1, col = 'blue', lty = 2)

plot(X_test[,1], rowMeans(test2$test_samples))
lines(X_test[1:101,1], u2_test_0, col = 'red')
lines(X_test[102:202,1], u2_test_1, col = 'red')
#sourceCpp("src/uni_bartFit.cpp")
sourceCpp("src/sep_bartFit.cpp")
source("scripts/makeCutpoints.R")




# uni_fit_1 <- uni_bartFit(Y = Y[,1], X = X_train, X_pred = X_train, cutpoints)
# uni_fit_2 <- uni_bartFit(Y = Y[,2], X = X_train, X_pred = X_train, cutpoints)

# Let's see how well plain vanilla BART works on this dataset
sep_fit <- sep_bartFit(Y = Y, X = X_train, X_pred = X_train, cutpoints)

sqrt( mean( (Y[,1] - rowMeans(sep_fit$fit_samples[,1,]))^2))
sqrt( mean( (Y[,2] - rowMeans(sep_fit$fit_samples[,2,]))^2))

png("images/toy_sepBART_fit.png", width = 6, height = 6, units = "in", res = 300)
par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0), mfrow = c(2,2), cex = 0.8, cex.main = 0.9, cex.lab = 0.8, cex.axis = 0.8)

plot(1, type = "n", xlim = range(rowMeans(sep_fit$fit_samples[,1,])), ylim = range(f1_train),
     xlab = expression(hat(f)[1]), ylab = expression(f[1]), main = "Actual vs Predicted -- Task 1")
points(rowMeans(sep_fit$fit_samples[,1,]), f1_train, pch = 16, cex = 0.4)
abline(a = 0, b = 1, col = 'green', lty = 2)

plot(1, type = "n", xlim = range(rowMeans(sep_fit$fit_samples[,2,])), ylim = range(f2_train),
     xlab = expression(hat(f)[2]), ylab = expression(f[2]), main = "Actual vs Predicted -- Task 2")
points(rowMeans(sep_fit$fit_samples[,2,]), f2_train, pch = 16, cex = 0.4)
abline(a = 0, b = 1, col = 'green', lty = 2)

plot(1, type = "n", xlim = c(1,1000), ylim = range(sqrt(sep_fit$Sigma_samples)), 
     xlab = "Iteration", ylab = expression(sigma[1]), main = "Residual standard deviation -- Task 1")
lines(1:1000, sqrt(sep_fit$Sigma_samples[1,1,]))
abline(h = sigma_1, col = "green")
plot(1, type = "n", xlim = c(1,1000), ylim = range(sqrt(sep_fit$Sigma_samples)),
     xlab = "Iteration", ylab = expression(sigma[,2]), main = "Residual standard deviation -- Task 2")
lines(1:1000, sqrt(sep_fit$Sigma_samples[2,2,]))
abline(h = sigma_2, col = 'green')
dev.off()
