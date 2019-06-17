# Test the new parametrization
library(Rcpp)
library(RcppArmadillo)

sourceCpp("src/slfm_BART_res.cpp")
sourceCpp("src/slfm_BART2.cpp")
sourceCpp("src/sep_BART.cpp")


source("scripts/makeCutpoints.R")
source("scripts/predictive_intervals.R")

load("~/Dropbox/Broderick_Group/bart_extensions/data/forex.RData")

res_1 <- slfm_BART_res(Y_train, X_train, X_train, cutpoints, D = 25, m_u = 10, m_h = 50)
res_2 <- slfm_BART_res(Y_train, X_train, X_train, cutpoints, D = 50, m_u = 10, m_h = 50)
res_3 <- slfm_BART_res(Y_train, X_train, X_train, cutpoints, D = 25, m_u = 50, m_h = 100)
res_4 <- slfm_BART_res(Y_train, X_train, X_train, cutpoints, D = 25, m_u = 50, m_h = 200)
res_5 <- slfm_BART_res(Y_train, X_train, X_train, cutpoints, D = 25, m_u = 50, m_h = 100)


slfm_D25_m50 <- slfm_BART2(Y_train, X_train, X_train, cutpoints, D = 25, m = 50)
slfm_D50_m100 <- slfm_BART2(Y_train, X_train, X_train, cutpoints, D = 50, m = 100)


sep_fit <- sep_BART(Y_train, X_train, X_train, cutpoints)

method_list <- c("res_1", "res_2", "res_3","res_4", "res_5", 
                 "slfm_D25_m50", "slfm_D50_m100", "sep_fit")
test_smse <- data.frame("USD.CAD" = rep(NA, times = length(method_list)),
                        "USD.JPY" = rep(NA, times = length(method_list)),
                        "USD.AUD" = rep(NA, times = length(method_list)))
rownames(test_smse) <- method_list
for(method in method_list){
  fit <- get(method)
  dimnames(fit$f_test_samples) <- list(c(), colnames(Y_train), c())
  #quant <- bart_quantiles(fit)
  #fit[["test_quantiles"]] <- quant
  for(x in colnames(test_smse)){
    test_index <- which(is.na(Y_train[,x]))
    test_smse[method,x] <- mean( (forex_raw[test_index,x] - rowMeans(fit$f_test_samples[test_index,x,]))^2)/var(Y_train[,x], na.rm = TRUE)
  }
  assign(method, fit)
}
test_smse[,"Avg"] <- rowMeans(test_smse)

par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0), mfrow = c(1,3), cex.lab = 0.8, cex.axis = 0.8, cex.main = 0.8)
for(x in c("USD.CAD", "USD.JPY", "USD.AUD")){
  y_lim <- range(c(forex_raw[,x],
                   rowMeans(slfm_D25_m50$f_test_samples[,x,]),
                   rowMeans(res_4$f_test_samples[,x,]),
                   rowMeans(sep_fit$f_test_samples[,x,])))
  plot(1, xlim = range(X_train), ylim = y_lim, type = "n", 
       xlab = "Time", ylab = "Exchange Rage", main = x)
  points(X_train, Y_train[,x], pch = 16, cex = 0.4)
  missing_ix <- which(is.na(Y_train[,x]))
  points(X_train[missing_ix], forex_raw[missing_ix,x], pch = 16, cex = 0.4, col = 'cyan')
  
  #lines(X_train, rowMeans(slfm2_m10_D50$f_test_samples[,x,]), col = 'green')
  lines(X_train, rowMeans(slfm_D25_m50$f_test_samples[,x,]), col = 'blue')
  lines(X_train, rowMeans(sep_fit$f_test_samples[,x,]), col = 'red')
  lines(X_train, rowMeans(res_4$f_test_samples[,x,]), col = 'green')
  
}


