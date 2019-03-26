# Script to run the forex experiment on supercloud
# Foreign exchange example
library(Rcpp)
library(RcppArmadillo)

sourceCpp("src/sep_bartFit.cpp")
sourceCpp("src/slfm_bart.cpp")

load("data/forex.RData")
m_list <- c(1, 5, 10, 25, 50, 100, 200)
#D_list <- c(5, 10, 25, 50, 100)
D_list <- c(5,10)

args <- commandArgs(TRUE_)
m <- m_list[as.numeric(args[1])]


method_list <- paste0("slfm_D", D_list, "_m", m)

train_smse <- matrix(nrow = length(method_list), ncol = ncol(Y_train), dimnames = list(method_list, colnames(Y_train)))
test_smse <- matrix(nrow = length(method_list), ncol = 3, dimnames = list(method_list, c("USD.CAD", "USD.JPY", "USD.AUD")))
test_smse2 <- matrix(nrow = length(method_list), ncol = 3, dimnames = list(method_list, c("USD.CAD", "USD.JPY", "USD.AUD")))

sigma <- matrix(nrow = length(method_list), ncol = ncol(Y_train), dimnames = list(method_list, colnames(Y_train)))
time <- rep(NA, times = length(D_list))
names(time) <- method_list

for(D_ix in 1:length(D_list)){
  D <- D_list[D_ix]
  print(paste("Starting D =", D, "at", Sys.time()))
  fit <- slfm_bartFit(Y_train, X_train, X_train, cutpoints, verbose = FALSE, D = D, m = m)
  dimnames(fit$f_test_samples) <- list(c(), colnames(Y_train), c())
  dimnames(fit$sigma_samples) <- list(colnames(Y_train),c())
  method <- method_list[D_ix]
  for(x in colnames(Y_train)){
    train_index <- which(!is.na(Y_train[,x]))
    train_smse[method, x] <- mean( (forex_raw[train_index,x] - rowMeans(fit$f_test_samples[train_index,x,]))^2)/var(Y_train[,x], na.rm = TRUE)
    sigma[method,x] <- mean(fit$sigma_samples[x,])
    time[x] <- fit$time
  }
  for(x in colnames(test_smse)){
    test_index <- which(is.na(Y_train[,x]))
    test_smse[method,x] <- mean( (forex_raw[test_index,x] - rowMeans(fit$f_test_samples[test_index,x,]))^2)/var(Y_train[,x], na.rm = TRUE)
    test_smse2[method,x] <- mean( (forex_raw[test_index,x] - rowMeans(fit$f_test_samples[test_index,x,]))^2)/var(forex_raw[test_index,x])
  }
}

train_smse <- as.data.frame(train_smse)
test_smse <- as.data.frame(test_smse)
test_smse2 <- as.data.frame(test_smse2)

train_smse[,"Avg"] <- rowMeans(train_smse)
test_smse[,"Avg"] <- rowMeans(test_smse)
test_smse2[,"Avg"] <- rowMeans(test_smse2)

assign(paste0("train_smse_m", m), train_smse)
assign(paste0("test_smse_m", m), test_smse)
assign(paste0("test_smse2_m", m), test_smse2)
assign(paste0("sigma_m", m), sigma)
assign(paste0("time_m",m), time)

save_list <- paste0(c("train_smse", "test_smse", "test_smse2", "sigma", "time"), "_m", m)
save(list = save_list, file = paste0("results/forex_m",m,".RData"))
