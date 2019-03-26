# Foreign exchange example
library(Rcpp)
library(RcppArmadillo)

sourceCpp("src/sep_bartFit.cpp")
sourceCpp("src/slfm_bart.cpp")


load("~/Dropbox/Broderick_Group/bart_extensions/data/forex.RData")


method_list <- c("slfm_D5_m100", "slfm_D10_m100", "slfm_D25_m100", "slfm_D50_m100", "slfm_D100_m100")

slfm_D5_m100 <- slfm_bartFit(Y_train, X_train, X_train, cutpoints, verbose = TRUE, D = 5, m = 100)
slfm_D10_m100 <- slfm_bartFit(Y_train, X_train, X_train, cutpoints, verbose = TRUE, D = 10, m = 100)
slfm_D25_m100 <- slfm_bartFit(Y_train, X_train, X_train, cutpoints, verbose = TRUE, D = 25, m = 100)
slfm_D50_m100 <- slfm_bartFit(Y_train, X_train, X_train, cutpoints, verbose = TRUE, D = 50, m = 100)
slfm_D100_m100 <- slfm_bartFit(Y_train, X_train, X_train, cutpoints, verbose = TRUE, D = 100, m = 100)


dimnames(slfm_D5_m100$f_test_samples) <- list(c(), colnames(Y_train), c())
dimnames(slfm_D10_m100$f_test_samples) <- list(c(), colnames(Y_train), c())
dimnames(slfm_D25_m100$f_test_samples) <- list(c(), colnames(Y_train), c())
dimnames(slfm_D50_m100$f_test_samples) <- list(c(), colnames(Y_train), c())
dimnames(slfm_D100_m100$f_test_samples) <- list(c(), colnames(Y_train), c())


#slfm_D10_m50 <- slfm_bartFit(Y_train, X_train, X_train, cutpoints, verbose = TRUE, D = 10, m = 50)

#slfm_D10_m200 <- slfm_bartFit(Y_train, X_train, X_train, cutpoints, verbose = TRUE, D = 10, m = 200)
#slfm_D25_m200 <- slfm_bartFit(Y_train, X_train, X_train, cutpoints, verbose = TRUE, D = 25, m = 50)

#dimnames(slfm_D10_m200$f_test_samples) <- list(c(), colnames(Y_train), c())
#dimnames(slfm_D25_m100$f_test_samples) <- list(c(), colnames(Y_train), c())
#dimnames(slfm_D10_m50$f_test_samples) <- list(c(), colnames(Y_train), c())
#dimnames(slfm_D50_m100$f_test_samples) <- list(c(), colnames(Y_train), c())
#dimnames(slfm_D100_m100$f_test_samples) <- list(c(), colnames(Y_train), c())

save(slfm_D10_m100, slfm_D25_m100, slfm_D50_m100, slfm_D100_m100, file = "~/Dropbox/Broderick_Group/bart_extensions/data/forex_slfm.RData")

# SMSE (pg 23 of Rasmussen and Williams): MSE/var(Y)

train_smse <- matrix(nrow = length(method_list), ncol = ncol(Y_train), dimnames = list(method_list, colnames(Y_train)))
test_smse <- matrix(nrow = length(method_list), ncol = 3, dimnames = list(method_list, c("USD.CAD", "USD.JPY", "USD.AUD")))


for(method in method_list){
  
  fit <- get(method)
  #test_smse <- rep(NA, times = 3)
  #names(test_smse) <- c("USD.CAD", "USD.JPY", "USD.AUD")
  #train_smse <- rep(NA, times = ncol(Y_train))
  #names(train_smse) <- colnames(Y_train)
  for(x in colnames(test_smse)){
    test_index <- which(is.na(Y_train[,x]))
    test_smse[method,x] <- mean( (forex_raw[test_index,x] - rowMeans(fit$f_test_samples[test_index,x,]))^2)/var(Y_train[,x], na.rm = TRUE)
  }
  for(x in colnames(train_smse)){
    train_index <- which(!is.na(Y_train[,x]))
    train_smse[method,x] <- mean( (forex_raw[train_index,x] - rowMeans(fit$f_test_samples[train_index,x,]))^2)/var(Y_train[,x], na.rm = TRUE)
  }
}
train_smse <- as.data.frame(train_smse)
test_smse <- as.data.frame(test_smse)

train_smse[,"Avg"] <- rowMeans(train_smse)
test_smse[,"Avg"] <- rowMeans(test_smse)

View(round(test_smse, digits = 4))

png("~/Dropbox/Broderick_Group/bart_extensions/images/forex_m100.png", width = 8.5, height = 8.5/3, units = "in", res = 300)
par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0), cex.main = 0.9, cex.lab = 0.8, cex.axis = 0.8, mfrow = c(1,3))
plot(1, type = "n", xlim = c(0, 250), ylim = c(0.75, 1.15), main = "CAD", ylab = "Exchange Rate", xlab = "Time (days)")
lines(rowMeans(slfm_D5_m100$f_test_samples[,"USD.CAD",]), col = 'black')
lines(rowMeans(slfm_D10_m100$f_test_samples[,"USD.CAD",]), col = 'green')
lines(rowMeans(slfm_D25_m100$f_test_samples[,"USD.CAD",]), col = 'purple')
lines(rowMeans(slfm_D50_m100$f_test_samples[,"USD.CAD",]), col = 'blue')
lines(rowMeans(slfm_D100_m100$f_test_samples[,"USD.CAD",]), col = 'red')
points(Y_train[,"USD.CAD"], pch = 16, cex = 0.5, col = 'pink')
points(50:100,forex_raw[50:100, "USD.CAD"], pch = 16, cex = 0.5, col = 'cyan')
legend("topleft", legend = paste("D =", c(5, 10, 25, 50, 100)), col = c('black', 'green', 'purple', 'blue', 'red'), lty = 1, bty = "n",
       cex = 0.7)
plot(1, type = "n", xlim = c(0, 250), ylim = c(7.8e-3, 9.4e-3), main = "JPY", ylab = "Exchange Rate", xlab = "Time (days)")
lines(rowMeans(slfm_D5_m100$f_test_samples[,"USD.JPY",]), col = 'black')
lines(rowMeans(slfm_D10_m100$f_test_samples[,"USD.JPY",]), col = 'green')
lines(rowMeans(slfm_D25_m100$f_test_samples[,"USD.JPY",]), col = 'purple')
lines(rowMeans(slfm_D50_m100$f_test_samples[,"USD.JPY",]), col = 'blue')
lines(rowMeans(slfm_D100_m100$f_test_samples[,"USD.JPY",]), col = 'red')
points(Y_train[,"USD.JPY"], pch = 16, cex = 0.5, col = 'pink')
points(100:150,forex_raw[100:150, "USD.JPY"], pch = 16, cex = 0.5, col = 'cyan')
legend("topleft", legend = paste("D =", c(5, 10, 25, 50, 100)), col = c('black', 'green', 'purple', 'blue', 'red'), lty = 1, bty = "n",
       cex = 0.7)
plot(1, type = "n", xlim = c(0, 250), ylim = c(0.7, 0.95), main = "AUD", ylab = "Exchange Rate", xlab = "Time (days)")
lines(rowMeans(slfm_D5_m100$f_test_samples[,"USD.AUD",]), col = 'black')
lines(rowMeans(slfm_D10_m100$f_test_samples[,"USD.AUD",]), col = 'green')
lines(rowMeans(slfm_D25_m100$f_test_samples[,"USD.AUD",]), col = 'purple')
lines(rowMeans(slfm_D50_m100$f_test_samples[,"USD.AUD",]), col = 'blue')
lines(rowMeans(slfm_D100_m100$f_test_samples[,"USD.AUD",]), col = 'red')
points(Y_train[,"USD.AUD"], pch = 16, cex = 0.5, col = 'pink')
points(150:200,forex_raw[150:200, "USD.AUD"], pch = 16, cex = 0.5, col = 'cyan')
legend("topleft", legend = paste("D =", c(5, 10, 25, 50, 100)), col = c('black', 'green', 'purple', 'blue', 'red'), lty = 1, bty = "n",
       cex = 0.7)
dev.off()
