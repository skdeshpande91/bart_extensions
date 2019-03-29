# Airtemperature example
library(Rcpp)
library(RcppArmadillo)

sourceCpp("src/sep_bartFit.cpp")
sourceCpp("src/slfm_bart.cpp")

source("scripts/predictive_intervals.R")

load("~/Dropbox/Broderick_Group/bart_extensions/data/airtemp.RData")

sep_fit <- sep_bartFit(Y, X, X, cutpoints, verbose = TRUE)
slfm_D5_m50 <- slfm_bartFit(Y, X, X, cutpoints, D = 5, m = 50, verbose = TRUE)

dimnames(sep_fit$f_test_samples) <- list(c(), colnames(Y), c())
dimnames(slfm_D5_m50$f_test_samples) <- list(c(), colnames(Y), c())

sep_quantiles <- bart_quantiles(sep_fit)
slfm_quantiles <- bart_quantiles(slfm_D5_m50)

sep_fit[["test_quantiles"]] <- sep_quantiles
slfm_D5_m50[["test_quantiles"]] <- slfm_quantiles

save(sep_fit, slfm_D5_m50, file = "~/Dropbox/Broderick_Group/bart_extensions/airtemp_example.RData")


method_list <- c("sep_fit", "slfm_D5_m50")
test_smse <- matrix(nrow = length(method_list), ncol = 2, dimnames = list(method_list, c("CAM", "CHI")))


for(method in method_list){
  fit <- get(paste0(method))
  for(x in colnames(test_smse)){
    test_index <- which(is.na(Y[,x]))
    test_smse[method,x] <- mean( (air_temp[test_index,x] - rowMeans(fit$f_test_samples[test_index,x,]))^2, na.rm = TRUE)/var(Y[,x], na.rm = TRUE)
  }
}


png("~/Dropbox/Broderick_Group/bart_extensions/images/airtemp_intervals.png", width = 2*8.5/3, height = 8.5/3, units = "in", res = 300)
par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0), cex.main = 0.9, cex.lab = 0.8, cex.axis = 0.8, mfrow = c(1,2))
plot(1, xlim = c(10, 15), ylim = c(5, 30), main = "Cambermet", ylab = "Temperature (C)", xlab = "Time (days)")

polygon(c(X, rev(X)), c(slfm_D5_m50$test_quantiles[,"CAM","0.025"], rev(slfm_D5_m50$test_quantiles[,"CAM","0.975"])), col = rgb(0,0,1,1/3), border = NA)
lines(X, rowMeans(slfm_D5_m50$f_test_samples[,"CAM",]), col = 'blue', lwd = 0.5)

polygon(c(X, rev(X)), c(sep_fit$test_quantiles[,"CAM","0.025"], rev(sep_fit$test_quantiles[,"CAM","0.975"])), col = rgb(1,0,0,1/3), border = NA)
lines(X, rowMeans(sep_fit$f_test_samples[,"CAM",]), col = 'red', lwd = 0.5)

points(X[which(is.na(Y[,"CAM"]))], air_temp[which(is.na(Y[,"CAM"])), "CAM"], pch = 16, cex= 0.1, col = 'green')
points(X, Y[,"CAM"], pch = 16, cex = 0.1, col = 'grey')
tmp_legend <- c(paste0("D = 5, m = 50, SMSE = ", round(test_smse["slfm_D5_m50","CAM"], digits = 4)),
                paste0("separate BART, SMSE = ", round(test_smse["sep_fit", "CAM"], digits = 4)))
legend("topleft", legend = tmp_legend, col = c("blue", "red"), lty = 1, bty = "n", cex = 0.4)

plot(1, xlim = c(10, 15), ylim = c(5, 30), main = "Chimet", ylab = "Temperature (C)", xlab = "Time (days)")

polygon(c(X, rev(X)), c(slfm_D5_m50$test_quantiles[,"CHI","0.025"], rev(slfm_D5_m50$test_quantiles[,"CHI","0.975"])), col = rgb(0,0,1,1/3), border = NA)
lines(X, rowMeans(slfm_D5_m50$f_test_samples[,"CHI",]), col = 'blue', lwd = 0.5)

polygon(c(X, rev(X)), c(sep_fit$test_quantiles[,"CHI","0.025"], rev(sep_fit$test_quantiles[,"CHI","0.975"])), col = rgb(1,0,0,1/3), border = NA)
lines(X, rowMeans(sep_fit$f_test_samples[,"CHI",]), col = 'red', lwd = 0.5)

points(X[which(is.na(Y[,"CHI"]))], air_temp[which(is.na(Y[,"CHI"])), "CHI"], pch = 16, cex= 0.1, col = 'green')
points(X, Y[,"CHI"], pch = 16, cex = 0.1, col = 'grey')
tmp_legend <- c(paste0("D = 5, m = 50, SMSE = ", round(test_smse["slfm_D5_m50","CHI"], digits = 4)),
                paste0("separate BART, SMSE = ", round(test_smse["sep_fit", "CHI"], digits = 4)))
legend("topleft", legend = tmp_legend, col = c("blue", "red"), lty = 1, bty = "n", cex = 0.4)
dev.off()






png("~/Dropbox/Broderick_Group/bart_extensions/images/airtemp.png", width = 2*8.5/3, height = 8.5/3, units = "in", res = 300)

par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0), cex.main = 0.9, cex.lab = 0.8, cex.axis = 0.8, mfrow = c(1,2))

plot(1, xlim = c(10, 15), ylim = c(5, 30), main = "Cambermet", ylab = "Temperature (C)", xlab = "Time (days)")
points(X[which(is.na(Y[,"CAM"]))], air_temp[which(is.na(Y[,"CAM"])),"CAM"], pch = 16, cex = 0.15, col = 'cyan')

lines(X, rowMeans(sep_fit$f_test_samples[,"CAM",]), col = 'red')
lines(X, rowMeans(slfm_D5_m50$f_test_samples[,"CAM",]), col = 'blue')
points(X, Y[,"CAM"], pch = 16, cex = 0.15, col = 'pink')

tmp_legend <- c(paste0("D = 5, m = 50, SMSE = ", round(test_smse["slfm_D5_m50","CAM"], digits = 4)),
                paste0("separate BART, SMSE = ", round(test_smse["sep_fit", "CAM"], digits = 4)))
legend("topleft", legend = tmp_legend, col = c("blue", "red"), lty = 1, bty = "n", cex = 0.4)

plot(1, xlim = c(10, 15), ylim = c(5, 30), main = "Chimet", ylab = "Temperature (C)", xlab = "Time (days)")
points(X[which(is.na(Y[,"CHI"]))], air_temp[which(is.na(Y[,"CHI"])),"CHI"], pch = 16, cex = 0.15, col = 'cyan')

lines(X, rowMeans(sep_fit$f_test_samples[,"CHI",]), col = 'red')
lines(X, rowMeans(slfm_D5_m50$f_test_samples[,"CHI",]), col = 'blue')
points(X, Y[,"CHI"], pch = 16, cex = 0.15, col = 'pink')

tmp_legend <- c(paste0("D = 5, m = 50, SMSE = ", round(test_smse["slfm_D5_m50","CHI"], digits = 4)),
                paste0("separate BART, SMSE = ", round(test_smse["sep_fit", "CHI"], digits = 4)))
legend("topleft", legend = tmp_legend, col = c("blue", "red"), lty = 1, bty = "n", cex = 0.4)
dev.off()
