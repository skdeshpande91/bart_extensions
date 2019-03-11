library(Rcpp)
library(RcppArmadillo)

#sourceCpp("src/uni_bartFit.cpp")
sourceCpp("src/sep_bartFit.cpp")
source("scripts/makeCutpoints.R")

load("data/toy_example.RData")

f_range <- c(-1,1) * max(abs(c(f1_full_0, f1_full_1, f2_full_0, f2_full_1)))
u_range <- c(-1,1) * max(abs(c(u1_full, u2_full_0, u2_full_1)))

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
