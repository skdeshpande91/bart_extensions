library(Rcpp)
library(RcppArmadillo)
sourceCpp("src/slfm_bart.cpp")

load("~/Dropbox/Broderick_Group/bart_extensions/data/airtemp.RData")

m_list <- c(1, 5, 10, 25, 50, 100, 200)
D_list <- c(5, 10, 25, 50)

args <- commandArgs(TRUE)
m <- m_list[as.numeric(args[1])]
method_list <- paste0("slfm_D", D_list, "_m", m)

train_smse <- matrix(nrow = length(method_list), ncol = ncol(Y), dimnames = list(method_list, colnames(Y)))
test_smse <- matrix(nrow = length(method_list), ncol = 2, dimnames = list(method_list, c("CAM", "CHI")))
test_smse2 <- matrix(nrow = length(method_list), ncol = 2, dimnames = list(method_list, c("CAM", "CHI")))

sigma <- matrix(nrow = length(method_list), ncol = ncol(Y), dimnames = list(method_list, colnames(Y)))
time <- rep(NA, times = length(D_list))
names(time) <- method_list

for(D_ix in 1:length(D_list)){
  D <- D_list[D_ix]
  print(paste("Starting D =", D, "at", Sys.time()))
  fit <- slfm_bartFit(Y, X, X, cutpoints, verbose = FALSE, D = D, m = m)
  dimnames(fit$f_test_samples) <- list(c(), colnames(Y), c())
  dimnames(fit$sigma_samples) <- list(colnames(Y),c())
  method <- method_list[D_ix]
  for(x in colnames(Y)){
    train_index <- which(!is.na(Y[,x]))
    train_smse[method, x] <- mean( (air_temp[train_index,x] - rowMeans(fit$f_test_samples[train_index,x,]))^2, na.rm = TRUE)/var(air_temp[,x], na.rm = TRUE)
    sigma[method,x] <- mean(fit$sigma_samples[x,])
    time[x] <- fit$time
  }
  
  test_smse[method, "CAM"] <- mean( (air_temp[cam_test_index,"CAM"] - rowMeans(fit$f_test_samples[cam_test_index,"CAM",]))^2, na.rm = TRUE)/var(air_temp[,"CAM"],na.rm = TRUE)
  test_smse2[method, "CAM"] <- mean( (air_temp[cam_test_index,"CAM"] - rowMeans(fit$f_test_samples[cam_test_index,"CAM",]))^2, na.rm = TRUE)/var(air_temp[cam_test_index,"CAM"],na.rm=TRUE)
  
  test_smse[method,"CHI"] <- mean( (air_temp[chi_test_index,"CHI"] - rowMeans(fit$f_test_samples[chi_test_index, "CHI",]))^2, na.rm = TRUE)/var(air_temp[,"CHI"], na.rm = TRUE)
  test_smse2[method, "CHI"] <- mean( (air_temp[chi_test_index, "CHI"] - rowMeans(fit$f_test_samples[chi_test_index, "CHI",]))^2, na.rm = TRUE)/var(air_temp[chi_test_index, "CHI"],na.rm = TRUE)
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
save(list = save_list, file = paste0("results/air_temp_m",m,".RData"))