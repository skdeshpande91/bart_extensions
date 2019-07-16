# new airtemp example

library(Rcpp)
library(RcppArmadillo)
sourceCpp("src/slfm_BART.cpp")
sourceCpp("src/slfm_BART2.cpp")
sourceCpp("src/slfm_BART3.cpp")
sourceCpp("src/sep_BART.cpp")

source("scripts/makeCutpoints.R")
source("scripts/predictive_intervals.R")
source("scripts/get_sigma_phi.R")

load("~/Dropbox/Broderick_Group/bart_extensions/data/airtemp.RData")

slfm_m50_D10 <- slfm_BART(Y, X, X, cutpoints, verbose = FALSE, D = 10, m = 50, nd = 1000, burn = 1500)
slfm_m50_D25 <- slfm_BART(Y, X, X, cutpoints, verbose = FALSE, D = 25, m = 50, nd = 1000, burn = 1500)
slfm_m10_D50 <- slfm_BART(Y, X, X, cutpoints, verbose = FALSE, D = 50, m = 10, nd = 1000, burn = 1500)

slfm2_m50_D10 <- slfm_BART2(Y, X, X, cutpoints, verbose = FALSE, D = 10, m = 50, nd = 1000, burn = 1500)
slfm2_m50_D25 <- slfm_BART2(Y, X, X, cutpoints, verbose = FALSE, D = 25, m = 50, nd = 1000, burn = 1500)
slfm2_m10_D50 <- slfm_BART2(Y, X, X, cutpoints, verbose = FALSE, D = 50, m = 10, nd = 1000, burn = 1500)

# Need to get the values of sigma_phi
sigma_phi_D10 <- rep(NA, times = ncol(Y))
sigma_phi_D25 <- rep(NA, times = ncol(Y))
sigma_phi_D50 <- rep(NA, times = ncol(Y))
for(k in 1:ncol(Y)){
  sigma_phi_D10[k] <- get_sigma_phi(Y[,k], target_prob = 0.8, df = 10)
  sigma_phi_D25[k] <- get_sigma_phi(Y[,k], target_prob = 0.8, df = 25)
  sigma_phi_D50[k] <- get_sigma_phi(Y[,k], target_prob = 0.8, df = 50)
}


slfm3_m50_D10 <- slfm_BART3(Y, X, X, cutpoints, sigma_phi = sigma_phi_D10, verbose = FALSE, D = 10, m = 50, nd = 1000, burn = 1500)
slfm3_m50_D25 <- slfm_BART3(Y, X, X, cutpoints, sigma_phi = sigma_phi_D25, verbose = FALSE, D = 25, m = 50, nd = 1000, burn = 1500)
slfm3_m10_D50 <- slfm_BART3(Y, X, X, cutpoints, sigma_phi = sigma_phi_D50, verbose = FALSE, D = 50, m = 10, nd = 1000, burn = 1500)

sep_fit <- sep_BART(Y, X, X, cutpoints, verbose = FALSE, nd = 1000, burn = 1500)

method_list <- c("slfm_m50_D10", "slfm2_m50_D10", "slfm3_m50_D10",
                 "slfm_m50_D25", "slfm2_m50_D25", "slfm3_m50_D25", 
                 "slfm_m10_D50", "slfm2_m10_D50", "slfm3_m10_D50", "sep_fit")
test_smse <- matrix(nrow = length(method_list), ncol = 2, dimnames = list(method_list, c("CAM", "CHI")))
for(method in method_list){
  fit <- get(method)
  dimnames(fit$f_test_samples) <- list(c(), colnames(Y), c())
  quant <- bart_quantiles(fit)
  fit[["test_quantiles"]] <- quant
  for(x in colnames(test_smse)){
    test_index <- which(is.na(Y[,x]))
    test_smse[method,x] <- mean( (air_temp[test_index,x] - rowMeans(fit$f_test_samples[test_index,x,]))^2, na.rm = TRUE)/var(Y[,x], na.rm = TRUE)
  }
  assign(method, fit)
}
test_smse <- cbind(test_smse, "Avg." = rowMeans(test_smse))

save(slfm_m50_D10, slfm2_m50_D10, slfm3_m50_D10, file = "~/Dropbox/Broderick_Group/bart_extensions/july16_airtemp_m50_D10.RData")
save(slfm_m50_D25, slfm2_m50_D25, slfm3_m50_D25, file = "~/Dropbox/Broderick_Group/bart_extensions/july16_airtemp_m50_D25.RData")
save(slfm_m10_D50, slfm2_m10_D50, slfm3_m10_D50, file = "~/Dropbox/Broderick_Group/bart_extensions/july16_airtemp_m10_D50.RData")
save(test_smse, file = "~/Dropbox/Broderick_Group/bart_extensions/july16_airtemp_test_smse.RData")

                   