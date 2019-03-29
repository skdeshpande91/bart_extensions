# Foreign exchange example
library(Rcpp)
library(RcppArmadillo)

sourceCpp("src/sep_bartFit.cpp")
sourceCpp("src/slfm_bart.cpp")
source("scripts/predictive_intervals.R")


load("~/Dropbox/Broderick_Group/bart_extensions/data/forex.RData")


sep_fit <- sep_bartFit(Y_train, X_train, X_train, cutpoints, verbose = TRUE)
slfm_D50_m50 <- slfm_bartFit(Y_train, X_train, X_train, cutpoints, verbose = TRUE, D = 50, m = 50)

dimnames(sep_fit$f_test_samples) <- list(c(), colnames(Y_train), c())
dimnames(slfm_D50_m50$f_test_samples) <- list(c(), colnames(Y_train), c())

sep_quantiles <- bart_quantiles(sep_fit)
slfm_quantiles <- bart_quantiles(slfm_D50_m50)


sep_fit[["test_quantiles"]] <- sep_quantiles
slfm_D50_m50[["test_quantiles"]] <- slfm_quantiles

method_list <- c("sep_fit", "slfm_D50_m50")

test_smse <- matrix(nrow = length(method_list), ncol = 3, dimnames = list(method_list, c("USD.CAD", "USD.JPY", "USD.AUD")))

for(method in method_list){
  fit <- get(paste0(method))
  for(x in colnames(test_smse)){
    test_index <- which(is.na(Y_train[,x]))
    test_smse[method,x] <- mean( (forex_raw[test_index,x] - rowMeans(fit$f_test_samples[test_index,x,]))^2)/var(Y_train[,x], na.rm = TRUE)
  }
}


# Plot the fits
png("~/Dropbox/Broderick_Group/bart_extensions/images/forex_intervals.png", width = 8.5, height = 8.5/3, units = "in", res = 300)
par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0), cex.main = 0.9, cex.lab = 0.8, cex.axis = 0.8, mfrow = c(1,3))
plot(1, type = "n", xlim = c(0, 250), ylim = c(0.75, 1.15), main = "CAD", ylab = "Exchange Rate", xlab = "Time (days)")

polygon(c(X_train, rev(X_train)), c(slfm_D50_m50$test_quantiles[,"USD.CAD","0.025"], rev(slfm_D50_m50$test_quantiles[,"USD.CAD", "0.975"])),
        col = rgb(0,0,1,1/3), border = NA)
polygon(c(X_train, rev(X_train)), c(slfm_D50_m50$test_quantiles[,"USD.CAD","0.025"], rev(sep_fit$test_quantiles[,"USD.CAD", "0.975"])),
        col = rgb(1,0,0,1/3), border = NA)
lines(rowMeans(sep_fit$f_test_samples[,"USD.CAD",]), col = 'red')
lines(rowMeans(slfm_D50_m50$f_test_samples[,"USD.CAD",]), col = 'blue')

points(X_train, Y_train[,"USD.CAD"], pch = 16, cex = 0.45, col = 'gray')
points(X_train[which(is.na(Y_train[,"USD.CAD"]))], forex_raw[which(is.na(Y_train[,"USD.CAD"])), "USD.CAD"], pch = 16, cex = 0.45, col = 'green')

tmp_legend <- c(paste0("D = 50, m = 50, SMSE = ", round(test_smse["slfm_D50_m50","USD.JPY"], digits = 4)),
                paste0("separate BART, SMSE = ", round(test_smse["sep_fit", "USD.JPY"], digits = 4)))

legend("topleft", legend = tmp_legend, col = c("blue", "red"), lty = 1, bty = "n", cex = 0.7)

plot(1, type = "n", xlim = c(0, 250), ylim = c(7.8e-3, 9.4e-3), main = "JPY", ylab = "Exchange Rate", xlab = "Time (days)")

polygon(c(X_train, rev(X_train)), c(slfm_D50_m50$test_quantiles[,"USD.JPY","0.025"], rev(slfm_D50_m50$test_quantiles[,"USD.JPY", "0.975"])),
        col = rgb(0,0,1,1/3), border = NA)
polygon(c(X_train, rev(X_train)), c(slfm_D50_m50$test_quantiles[,"USD.JPY","0.025"], rev(sep_fit$test_quantiles[,"USD.JPY", "0.975"])),
        col = rgb(1,0,0,1/3), border = NA)
lines(rowMeans(sep_fit$f_test_samples[,"USD.JPY",]), col = 'red')
lines(rowMeans(slfm_D50_m50$f_test_samples[,"USD.JPY",]), col = 'blue')

points(X_train, Y_train[,"USD.JPY"], pch = 16, cex = 0.45, col = 'gray')
points(X_train[which(is.na(Y_train[,"USD.JPY"]))], forex_raw[which(is.na(Y_train[,"USD.JPY"])), "USD.JPY"], pch = 16, cex = 0.45, col = 'green')


tmp_legend <- c(paste0("D = 50, m = 50, SMSE = ", round(test_smse["slfm_D50_m50","USD.JPY"], digits = 4)),
                paste0("separate BART, SMSE = ", round(test_smse["sep_fit", "USD.JPY"], digits = 4)))

legend("topleft", legend = tmp_legend, col = c("blue", "red"), lty = 1, bty = "n", cex = 0.7)

plot(1, type = "n", xlim = c(0, 250), ylim = c(0.7, 0.95), main = "AUD", ylab = "Exchange Rate", xlab = "Time (days)")

polygon(c(X_train, rev(X_train)), c(slfm_D50_m50$test_quantiles[,"USD.AUD","0.025"], rev(slfm_D50_m50$test_quantiles[,"USD.AUD", "0.975"])),
        col = rgb(0,0,1,1/3), border = NA)
polygon(c(X_train, rev(X_train)), c(slfm_D50_m50$test_quantiles[,"USD.AUD","0.025"], rev(sep_fit$test_quantiles[,"USD.AUD", "0.975"])),
        col = rgb(1,0,0,1/3), border = NA)
lines(rowMeans(sep_fit$f_test_samples[,"USD.AUD",]), col = 'red')
lines(rowMeans(slfm_D50_m50$f_test_samples[,"USD.AUD",]), col = 'blue')

points(X_train, Y_train[,"USD.AUD"], pch = 16, cex = 0.45, col = 'gray')
points(X_train[which(is.na(Y_train[,"USD.AUD"]))], forex_raw[which(is.na(Y_train[,"USD.AUD"])), "USD.AUD"], pch = 16, cex = 0.45, col = 'green')

tmp_legend <- c(paste0("D = 50, m = 50, SMSE = ", round(test_smse["slfm_D50_m50","USD.AUD"], digits = 4)),
                paste0("separate BART, SMSE = ", round(test_smse["sep_fit", "USD.AUD"], digits = 4)))

legend("topleft", legend = tmp_legend, col = c("green", "red"), lty = 1, bty = "n", cex = 0.7)
dev.off()




png("~/Dropbox/Broderick_Group/bart_extensions/images/forex.png", width = 8.5, height = 8.5/3, units = "in", res = 300)
par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0), cex.main = 0.9, cex.lab = 0.8, cex.axis = 0.8, mfrow = c(1,3))
plot(1, type = "n", xlim = c(0, 250), ylim = c(0.75, 1.15), main = "CAD", ylab = "Exchange Rate", xlab = "Time (days)")
lines(rowMeans(sep_fit$f_test_samples[,"USD.CAD",]), col = 'red')
lines(rowMeans(slfm_D50_m50$f_test_samples[,"USD.CAD",]), col = 'green')
points(50:100, forex_raw[50:100, "USD.CAD"], pch= 16, col = 'cyan', cex = 0.5)
points(Y_train[,"USD.CAD"], pch = 16, cex = 0.5, col = 'pink')

tmp_legend <- c(paste0("D = 50, m = 50, SMSE = ", round(test_smse["slfm_D50_m50","USD.CAD"], digits = 4)),
                paste0("separate BART, SMSE = ", round(test_smse["sep_fit", "USD.CAD"], digits = 4)))

legend("topleft", legend = tmp_legend, col = c("green", "red"), lty = 1, bty = "n", cex = 0.7)


plot(1, type = "n", xlim = c(0, 250), ylim = c(7.8e-3, 9.4e-3), main = "JPY", ylab = "Exchange Rate", xlab = "Time (days)")
lines(rowMeans(sep_fit$f_test_samples[,"USD.JPY",]), col = 'red')
lines(rowMeans(slfm_D50_m50$f_test_samples[,"USD.JPY",]), col = 'green')
points(100:150, forex_raw[100:150, "USD.JPY"], pch= 16, col = 'cyan', cex = 0.5)
points(Y_train[,"USD.JPY"], pch = 16, cex = 0.5, col = 'pink')
tmp_legend <- c(paste0("D = 50, m = 50, SMSE = ", round(test_smse["slfm_D50_m50","USD.JPY"], digits = 4)),
                paste0("separate BART, SMSE = ", round(test_smse["sep_fit", "USD.JPY"], digits = 4)))

legend("topleft", legend = tmp_legend, col = c("green", "red"), lty = 1, bty = "n", cex = 0.7)
plot(1, type = "n", xlim = c(0, 250), ylim = c(0.7, 0.95), main = "AUD", ylab = "Exchange Rate", xlab = "Time (days)")
lines(rowMeans(sep_fit$f_test_samples[,"USD.AUD",]), col = 'red')
lines(rowMeans(slfm_D50_m50$f_test_samples[,"USD.AUD",]), col = 'green')
points(150:200, forex_raw[150:200, "USD.AUD"], pch= 16, col = 'cyan', cex = 0.5)
points(Y_train[,"USD.AUD"], pch = 16, cex = 0.5, col = 'pink')
tmp_legend <- c(paste0("D = 50, m = 50, SMSE = ", round(test_smse["slfm_D50_m50","USD.AUD"], digits = 4)),
                paste0("separate BART, SMSE = ", round(test_smse["sep_fit", "USD.AUD"], digits = 4)))

legend("topleft", legend = tmp_legend, col = c("green", "red"), lty = 1, bty = "n", cex = 0.7)
dev.off()