library(Rcpp)
library(RcppArmadillo)

sourceCpp("src/slfm_BART.cpp")
sourceCpp("src/sep_BART.cpp")
sourceCpp("src/univariate_BART.cpp")


source("scripts/makeCutpoints.R")
source("scripts/predictive_intervals.R")

load("~/Dropbox/Broderick_Group/bart_extensions/data/forex.RData")


slfm_m50_D50 <- slfm_BART(Y_train, X_train, X_train, cutpoints, verbose = FALSE, D = 50, m = 50)
slfm_m50_D10 <- slfm_BART(Y_train, X_train, X_train, cutpoints, verbose = FALSE, D = 10, m = 50)
slfm_m25_D50 <- slfm_BART(Y_train, X_train, X_train, cutpoints, verbose = FALSE, D = 50, m = 25)
slfm_m25_D10 <- slfm_BART(Y_train, X_train, X_train, cutpoints, verbose = FALSE, D = 10, m = 25)

sep_fit1 <- sep_BART(Y_train, X_train, X_train, cutpoints, burn = 1000)
sep_fit2 <- sep_BART(Y_train, X_train, X_train, cutpoints, burn = 1000)

uni_fit1 <- univariate_BART(Y_train[,"USD.JPY"], X_train, X_train, cutpoints, burn = 1000)
uni_fit2 <- univariate_BART(Y_train[,"USD.JPY"], X_train, X_train, cutpoints, burn = 1000)


dimnames(slfm_m50_D50$f_test_samples) <- list(c(), colnames(Y_train), c())
dimnames(slfm_m50_D10$f_test_samples) <- list(c(), colnames(Y_train), c())
dimnames(slfm_m25_D50$f_test_samples) <- list(c(), colnames(Y_train), c())
dimnames(slfm_m25_D10$f_test_samples) <- list(c(), colnames(Y_train), c())


dimnames(sep_fit1$f_test_samples) <- list(c(), colnames(Y_train), c())
dimnames(sep_fit2$f_test_samples) <- list(c(), colnames(Y_train), c())


plot(X_train, rowMeans(sep_fit1$f_test_samples[,"USD.JPY",]), type = "l", ylim = c(0.0078, 0.0092))
lines(X_train, rowMeans(sep_fit2$f_test_samples[,"USD.JPY",]), col = 'red')
lines(X_train, rowMeans(uni_fit1$f_test_samples), col = 'blue')
lines(X_train, rowMeans(uni_fit2$f_test_samples), col = 'green')


method_list <- c("slfm_m50_D50", "slfm_m50_D10", "slfm_m25_D50", "slfm_m25_D10", "sep_fit1", "sep_fit2")
test_smse <- data.frame("USD.CAD" = rep(NA, times = length(method_list)),
                        "USD.JPY" = rep(NA, times = length(method_list)),
                        "USD.AUD" = rep(NA, times = length(method_list)))
rownames(test_smse) <- method_list
for(method in method_list){
  fit <- get(method)
  quant <- bart_quantiles(fit)
  fit[["test_quantiles"]] <- quant
  for(x in colnames(test_smse)){
    test_index <- which(is.na(Y_train[,x]))
    test_smse[method,x] <- mean( (forex_raw[test_index,x] - rowMeans(fit$f_test_samples[test_index,x,]))^2)/var(Y_train[,x], na.rm = TRUE)
  }
  assign(method, fit)
}
test_smse[,"Avg."] <- rowMeans(test_smse)



test_smse <- matrix(nrow = length(method_list), ncol = 3, dimnames = list(method_list, c("USD.CAD", "USD.JPY", "USD.AUD")))
for(method in method_list){
  fit <- get(paste0(method))
  for(x in colnames(test_smse)){
    test_index <- which(is.na(Y_train[,x]))
    test_smse[method,x] <- mean( (forex_raw[test_index,x] - rowMeans(fit$f_test_samples[test_index,x,]))^2)/var(Y_train[,x], na.rm = TRUE)
  }
}


sep_quantiles <- bart_quantiles(sep_fit)
slfm_quantiles <- bart_quantiles(slfm_D50_m50)

new_slfm <- slfm_BART(Y_train, X_train, X_train, cutpoints, verbose = TRUE, D = 10, m = 50)




plot(X_train, Y_train[,6], pch = 16, cex = 0.5, ylim = range(rowMeans(new_slfm$f_test_samples[,6,])))
lines(X_train, rowMeans(new_slfm$f_test_samples[,6,]), col = 'red')
points(X_train, forex_raw[,"USD.JPY"], cex = 0.4, pch = 4)
