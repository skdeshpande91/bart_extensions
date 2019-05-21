library(Rcpp)
library(RcppArmadillo)

sourceCpp("src/slfm_BART.cpp")
sourceCpp("src/sep_BART.cpp")
sourceCpp("src/uni_bartFit.cpp")
sourceCpp("src/univariate_BART.cpp")


source("scripts/predictive_intervals.R")


load("~/Dropbox/Broderick_Group/bart_extensions/data/forex.RData")

# check univariate BART

test_jpy <- univariate_BART(Y_train[,"USD.JPY"], X_train, X_train, cutpoints)




slfm_fit <- slfm_BART(Y_train, X_train, X_train, cutpoints, weight = 1, burn = 500, nd = 1000, D = 50, m = 25, verbose = TRUE)
sep_fit <- sep_BART(Y_train, X_train, X_train, cutpoints, nd = 1000, burn = 250)

dimnames(slfm_fit$f_test_samples) <- list(c(), colnames(Y_train), c())
dimnames(sep_fit$f_test_samples) <- list(c(), colnames(Y_train), c())

method_list <- c("slfm_fit", "sep_fit")

test_smse <- matrix(nrow = length(method_list), ncol = 3, dimnames = list(method_list, c("USD.CAD", "USD.JPY", "USD.AUD")))


for(method in method_list){
  fit <- get(paste0(method))
  for(x in colnames(test_smse)){
    test_index <- which(is.na(Y_train[,x]))
    test_smse[method,x] <- mean( (forex_raw[test_index,x] - rowMeans(fit$f_test_samples[test_index,x,]))^2)/var(Y_train[,x], na.rm = TRUE)
  }
}
