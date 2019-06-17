library(Rcpp)
library(RcppArmadillo)

sourceCpp("src/slfm_BART.cpp")
sourceCpp("src/slfm_BART2.cpp")

sourceCpp("src/slfm_BART_res.cpp")

sourceCpp("src/sep_BART.cpp")
sourceCpp("src/univariate_BART.cpp")


#sourceCpp("src/sep_bartFit.cpp")

source("scripts/makeCutpoints.R")
source("scripts/predictive_intervals.R")

load("~/Dropbox/Broderick_Group/bart_extensions/data/forex.RData")

########## June 17 simulations ########

slfm_m50_D10 <- slfm_BART(Y_train, X_train, X_train, cutpoints, verbose = FALSE, D = 10, m = 50)
slfm_m50_D25 <- slfm_BART(Y_train, X_train, X_train, cutpoints, verbose = FALSE, D = 25, m = 50)
slfm_m10_D50 <- slfm_BART(Y_train, X_train, X_train, cutpoints, verbose = FALSE, D = 50, m = 10)

slfm2_m50_D10 <- slfm_BART2(Y_train, X_train, X_train, cutpoints, verbose = FALSE, D = 10, m = 50)
slfm2_m50_D25 <- slfm_BART2(Y_train, X_train, X_train, cutpoints, verbose = FALSE, D = 25, m = 50)
slfm2_m10_D50 <- slfm_BART2(Y_train, X_train, X_train, cutpoints, verbose = FALSE, D = 50, m = 10)

sep_fit <- sep_BART(Y_train, X_train, X_train, cutpoints, verbose = FALSE)


test1 <- slfm_BART_res(Y_train, X_train, X_train, cutpoints, D = 50, m_u = 10, m_h = 50)
test2 <- slfm_BART_res(Y_train, X_train, X_train, cutpoints, D = 25, m_u = 10, m_h = 200)

#dimnames(slfm_m50_D10$f_test_samples) <- list(c(), colnames(Y_train), c())
#dimnames(slfm2_m50_D10$f_test_samples) <- list(c(), colnames(Y_train), c())
#dimnames(slfm_m50_D25$f_test_samples) <- list(c(), colnames(Y_train), c())
#dimnames(slfm2_m50_D25$f_test_samples) <- list(c(), colnames(Y_train), c())
#dimnames(slfm_m10_D50$f_test_samples) <- list(c(), colnames(Y_train), c())
#dimnames(slfm2_m10_D50$f_test_samples) <- list(c(), colnames(Y_train), c())

#dimnames(sep_fit$f_test_samples) <- list(c(), colnames(Y_train),c())



method_list <- c("slfm_m50_D10", "slfm2_m50_D10", "slfm_m50_D25", "slfm2_m50_D25", "slfm_m10_D50", "slfm2_m10_D50", "sep_fit", "test_new")
test_smse <- data.frame("USD.CAD" = rep(NA, times = length(method_list)),
                        "USD.JPY" = rep(NA, times = length(method_list)),
                        "USD.AUD" = rep(NA, times = length(method_list)))
rownames(test_smse) <- method_list
for(method in method_list){
  fit <- get(method)
  dimnames(fit$f_test_samples) <- list(c(), colnames(Y_train), c())
  quant <- bart_quantiles(fit)
  fit[["test_quantiles"]] <- quant
  for(x in colnames(test_smse)){
    test_index <- which(is.na(Y_train[,x]))
    test_smse[method,x] <- mean( (forex_raw[test_index,x] - rowMeans(fit$f_test_samples[test_index,x,]))^2)/var(Y_train[,x], na.rm = TRUE)
  }
  assign(method, fit)
}
test_smse[,"Avg."] <- rowMeans(test_smse)

par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0), mfrow = c(1,3), cex.lab = 0.8, cex.axis = 0.8, cex.main = 0.8)
for(x in c("USD.CAD", "USD.JPY", "USD.AUD")){
  y_lim <- range(c(forex_raw[,x],
                   rowMeans(slfm_m50_D10$f_test_samples[,x,]),
                   rowMeans(slfm_m50_D25$f_test_samples[,x,]),
                   rowMeans(slfm_m10_D50$f_test_samples[,x,]),
                   rowMeans(slfm2_m50_D10$f_test_samples[,x,]),
                   rowMeans(slfm2_m50_D25$f_test_samples[,x,]),
                   rowMeans(slfm2_m10_D50$f_test_samples[,x,]),
                   rowMeans(sep_fit$f_test_samples[,x,]),
                   rowMeans(test_new$f_test_samples[,x,])))
  
  plot(1, xlim = range(X_train), ylim = y_lim, type = "n", 
       xlab = "Time", ylab = "Exchange Rage", main = x)
  points(X_train, Y_train[,x], pch = 16, cex = 0.4)
  missing_ix <- which(is.na(Y_train[,x]))
  points(X_train[missing_ix], forex_raw[missing_ix,x], pch = 16, cex = 0.4, col = 'cyan')
  
  #lines(X_train, rowMeans(slfm2_m10_D50$f_test_samples[,x,]), col = 'green')
  lines(X_train, rowMeans(slfm2_m50_D25$f_test_samples[,x,]), col = 'blue')
  lines(X_train, rowMeans(sep_fit$f_test_samples[,x,]), col = 'red')
  
  lines(X_train, rowMeans(test_new$f_test_samples[,x,]), col = 'green')
  
}




plot(X_train, Y_try_ain[,"USD.CAD"], pch = 16, cex = 0.4)
points(X_train, forex_raw[,"USD.CAD"], pch = 16, cex = 0.5, col = 'blue')
lines(X_train, rowMeans(slfm2_m50_D10$f_test_samples[,"USD.CAD",]), col = 'green')
lines(X_train, rowMeans(slfm_m50_D10$f_test_samples[,"USD.CAD",]), col = 'red')
lines(X_train, rowMeans(slfm2_m50_D25$f_test_samples[,"USD.CAD",]), col = 'green', lty = 2)



plot(X_train, rowMeans(sep_fit1$f_test_samples[,"USD.JPY",]), type = "l", ylim = c(0.007, 0.01))
lines(X_train, rowMeans(sep_fit2$f_test_samples[,"USD.JPY",]), col = 'red')
lines(X_train, rowMeans(uni_fit1$f_test_samples), col = 'blue')
lines(X_train, rowMeans(uni_fit2$f_test_samples), col = 'green')
lines(X_train, rowMeans(sep_fit_old$f_test_samples[,"USD.JPY",]), col = 'orange')
lines(X_train, rowMeans(sep_fit_old2$f_test_samples[,"USD.JPY",]), col = 'purple')

########## old code is below #########

slfm_m50_D50 <- slfm_BART(Y_train, X_train, X_train, cutpoints, verbose = FALSE, D = 50, m = 50)
slfm_m50_D10 <- slfm_BART(Y_train, X_train, X_train, cutpoints, verbose = FALSE, D = 10, m = 50)
slfm_m25_D50 <- slfm_BART(Y_train, X_train, X_train, cutpoints, verbose = FALSE, D = 50, m = 25)
slfm_m25_D10 <- slfm_BART(Y_train, X_train, X_train, cutpoints, verbose = FALSE, D = 10, m = 25)

sep_fit1 <- sep_BART(Y_train, X_train, X_train, cutpoints, burn = 1000)
sep_fit2 <- sep_BART(Y_train, X_train, X_train, cutpoints, burn = 1000)


sep_fit_old <- sep_bartFit(Y_train, X_train, X_train, cutpoints)
sep_fit_old2 <- sep_bartFit(Y_train, X_train, X_train, cutpoints)




uni_fit1 <- univariate_BART(Y_train[,"USD.JPY"], X_train, X_train, cutpoints, burn = 1000)
uni_fit2 <- univariate_BART(Y_train[,"USD.JPY"], X_train, X_train, cutpoints, burn = 1000)


dimnames(slfm_m50_D50$f_test_samples) <- list(c(), colnames(Y_train), c())
dimnames(slfm_m50_D10$f_test_samples) <- list(c(), colnames(Y_train), c())
dimnames(slfm_m25_D50$f_test_samples) <- list(c(), colnames(Y_train), c())
dimnames(slfm_m25_D10$f_test_samples) <- list(c(), colnames(Y_train), c())


dimnames(sep_fit1$f_test_samples) <- list(c(), colnames(Y_train), c())
dimnames(sep_fit2$f_test_samples) <- list(c(), colnames(Y_train), c())

dimnames(sep_fit_old$f_test_samples) <- list(c(), colnames(Y_train), c())
dimnames(sep_fit_old2$f_test_samples) <- list(c(), colnames(Y_train), c())


plot(X_train, rowMeans(sep_fit1$f_test_samples[,"USD.JPY",]), type = "l", ylim = c(0.007, 0.01))
lines(X_train, rowMeans(sep_fit2$f_test_samples[,"USD.JPY",]), col = 'red')
lines(X_train, rowMeans(uni_fit1$f_test_samples), col = 'blue')
lines(X_train, rowMeans(uni_fit2$f_test_samples), col = 'green')
lines(X_train, rowMeans(sep_fit_old$f_test_samples[,"USD.JPY",]), col = 'orange')
lines(X_train, rowMeans(sep_fit_old2$f_test_samples[,"USD.JPY",]), col = 'purple')



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
