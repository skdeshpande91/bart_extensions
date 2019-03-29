# Code for the toy2 data.

library(Rcpp)
library(RcppArmadillo)

sourceCpp("src/sep_bartFit.cpp")
sourceCpp("src/slfm_bart.cpp")

source("scripts/makeCutpoints.R")
load("data/toy_example2.RData")
tmp_Y <- Y
cutpoints <- makeCutpoints(X_train)

args <- commandArgs(TRUE)
batch <- as.numeric(args[1])
m <- as.numeric(args[2]) # number of trees


q <- 10
n <- 1000
batch <- 1
m <- 100

tmp_results <- matrix(nrow = 10, ncol = 3*q + 1, dimnames = list(c(), c(paste0("Task", 1:10, "_train"), paste0("Task", 1:10, "_test"), paste0("sigma", 1:10), "Time")))
results <- list()
for(method in c("sep", "slfm_D2", "slfm_D5", "slfm_D10", "slfm_D20", "slfm_D50")){
  results[[paste0(method,"_m", m)]] <- tmp_results
}

for(r in 1:2){
  print(paste("Starting r = ", r, "at", Sys.time()))
  set.seed(12991 + 10*(batch-1) + r)
  for(k in 1:q){
    tmp_train <- get(paste0("f", k, "_train"))
    tmp_test <- get(paste0("f", k, "_test"))
    tmp_Y[,k] <- tmp_train + sigma[k]*rnorm(n, 0, 1)
  }
  
  sep <- sep_bartFit(tmp_Y, X_train, X_test, cutpoints, burn = 5, nd = 10, m = m)
  print("  Finished sep_BART")
  slfm_D2 <- slfm_bartFit(tmp_Y, X_train, X_test, cutpoints, burn = 5, nd = 10, D = 2, m = m)
  print("  Finished slfm_D2")
  slfm_D5 <- slfm_bartFit(tmp_Y, X_train, X_test, cutpoints, burn = 5, nd = 10, D = 5, m = m)
  print("  Finished slfm_D5")
  slfm_D10 <- slfm_bartFit(tmp_Y, X_train, X_test, cutpoints, burn = 5, nd = 10, D = 10, m = m)
  print("  Finished slfm_D10")
  slfm_D20 <- slfm_bartFit(tmp_Y, X_train, X_test, cutpoints, burn = 5, nd = 10, D = 20, m = m)
  print("  Finished slfm_D20")
  slfm_D50 <- slfm_bartFit(tmp_Y, X_train, X_test, cutpoints, burn = 5, nd = 10, D = 50, m = m)
  print("  Finished slfm_D50")
  
  # write the results
  for(method in c("sep", "slfm_D2", "slfm_D5", "slfm_D10", "slfm_D20", "slfm_D50")){
    tmp_fit <- get(method)
    for(k in 1:q){
      tmp_train <- get(paste0("f", k, "_train"))
      tmp_test <- get(paste0("f", k, "_test"))
      results[[paste0(method, "_m", m)]][r, paste0("Task", k, "_train")] <- sqrt( mean( (tmp_train - rowMeans(tmp_fit$f_train_samples[,k,]))^2))
      results[[paste0(method, "_m", m)]][r, paste0("Task", k, "_test")] <- sqrt( mean( (tmp_test - rowMeans(tmp_fit$f_test_samples[,k,]))^2))
      results[[paste0(method, "_m",m)]][r, paste0("sigma", k)] <- mean(sqrt(tmp_fit$sigma_samples[k,]))
      results[[paste0(method, "_m",m)]][r,"Time"] <- tmp_fit$time
    }
  }
  
}
assign(paste0("results_m", m, "_", batch), results)
save(list = paste0("results_m", m, "_", batch), file = paste0("results/toy2_m", m, "_", batch, ".RData"))

