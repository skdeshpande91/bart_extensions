# prepare the Jure dataset
library(Rcpp)
library(RcppArmadillo)
sourceCpp("src/slfm_bart.cpp")
source("scripts/makeCutpoints.R")

jura_train_raw <- read.table(file = "~/Dropbox/Broderick_Group/bart_extensions/data/jura_sample.dat", header = FALSE)
jura_test_raw <- read.table(file = "~/Dropbox/Broderick_Group/bart_extensions/data/jura_validation.dat",header = TRUE)

n_train <- nrow(jura_train_raw)
n_test <- nrow(jura_test_raw)
colnames(jura_train_raw) <- colnames(jura_test_raw)

jura_all <- rbind(jura_train_raw, jura_test_raw)

X_train <- as.matrix(jura_all[,c("Xloc", "Yloc")], ncol = 2)
Y_train <- as.matrix(jura_all[,c("Cd", "Ni", "Zn")])
Y_train[(n_train+1):(n_train+n_test),"Cd"] <- NA

cutpoints <- makeCutpoints(X_train)

slfm_D200_m1 <- slfm_bartFit(Y_train, X_train, X_train, cutpoints, D = 200, m = 1, verbose = FALSE)
slfm_D100_m10 <- slfm_bartFit(Y_train, X_train, X_train, cutpoints, D = 100, m = 10, verbose = FALSE)
slfm_D10_m50 <- slfm_bartFit(Y_train ,X_train, X_train, cutpoints, D = 10, m = 50, verbose = FALSE)
slfm_D10_m100 <- slfm_bartFit(Y_train, X_train, X_train, cutpoints, D = 10, m = 100, verbose = FALSE)
slfm_D10_m200 <- slfm_bartFit(Y_train, X_train, X_train, cutpoints, D = 10, m = 200, verbose = FALSE)
slfm_D25_m100 <- slfm_bartFit(Y_train, X_train, X_train, cutpoints, D = 25, m = 100, verbose = FALSE)

dimnames(slfm_D200_m1$f_test_samples) <- list(c(), colnames(Y_train), c())
dimnames(slfm_D100_m10$f_test_samples) <- list(c(), colnames(Y_train), c())
dimnames(slfm_D10_m50$f_test_samples) <- list(c(), colnames(Y_train), c())
dimnames(slfm_D10_m100$f_test_samples) <- list(c(), colnames(Y_train), c())
dimnames(slfm_D10_m200$f_test_samples) <- list(c(), colnames(Y_train), c())
dimnames(slfm_D25_m100$f_test_samples) <- list(c(), colnames(Y_train), c())
test_index <- which(is.na(Y_train[,"Cd"]))
# MAE
mean(abs(jura_all[test_index,"Cd"] - rowMeans(slfm_D200_m1$f_test_samples[test_index,"Cd",])))
mean(abs(jura_all[test_index,"Cd"] - rowMeans(slfm_D100_m10$f_test_samples[test_index,"Cd",])))
mean(abs(jura_all[test_index,"Cd"] - rowMeans(slfm_D10_m50$f_test_samples[test_index,"Cd",])))
mean(abs(jura_all[test_index,"Cd"] - rowMeans(slfm_D10_m100$f_test_samples[test_index,"Cd",])))
mean(abs(jura_all[test_index,"Cd"] - rowMeans(slfm_D10_m200$f_test_samples[test_index, "Cd",])))
mean(abs(jura_all[test_index,"Cd"] - rowMeans(slfm_D25_m100$f_test_samples[test_index,"Cd",])))


# In all cases the MAE is above 0.5, which is quite a bit away from the results in Table 3.6.1 in Andrew Gordon Wilson's PhD Thesis
# Also in GPAR, all of the state-of-the-art values are around 0.4
# I think this is due mostly to BART relying on axis-aligned splits
