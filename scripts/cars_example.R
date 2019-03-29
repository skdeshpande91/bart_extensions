library(Rcpp)
library(RcppArmadillo)

sourceCpp("src/sep_bartFit.cpp")
sourceCpp("src/slfm_bart.cpp")
source("scripts/makeCutpoints.R")

cars <- read.csv(file = "data/uci_automobile.csv", stringsAsFactors = TRUE)
# we will try to predict city.mpg, highway.mpg, and price using all of the other ones

# remove the rows with missing data
missing.index <- which(rowSums(is.na(cars)) != 0)
cars <- cars[-missing.index,]

# Get the model matrix

tmp <- lm(price ~ ., data = cars[,!colnames(cars) %in% c("city.mpg", "highway.mpg")], x = TRUE)
X_all <- tmp$x


Y_all <- as.matrix(cars[,c("city.mpg", "highway.mpg", "price")])

set.seed(32519)
n_all <- nrow(Y_all)
n_train <- ceiling(2/3 * n_all)
n_test <- n_all - n_train

train_index <- sample(1:n_all)[1:n_train]
test_index <- (1:n_all)[!(1:n_all) %in% train_index]
X_train <- X_all[train_index,]
X_test <- X_all[test_index,]
Y_train <- Y_all[train_index,]
Y_test <- Y_all[test_index,]

cutpoints <- makeCutpoints(X_all, gridlen = 10000)
q <- ncol(Y)

train_results <- matrix(nrow = 5, ncol = 3, dimnames = list(c("sep", "slfm_D5", "slfm_D10", "slfm_D25", "slfm_D50"), c("city.mpg", "highway.mpg", "price")))
test_results <- matrix(nrow = 5, ncol = 3, dimnames = list(c("sep", "slfm_D5", "slfm_D10", "slfm_D25", "slfm_D50"), c("city.mpg", "highway.mpg", "price")))
sigma <- matrix(nrow = 5, ncol = 3, dimnames = list(c("sep", "slfm_D5", "slfm_D10", "slfm_D25", "slfm_D50"), c("city.mpg", "highway.mpg", "price")))


results <- list()

method_list <- paste0(c("sep", "slfm_D5", "slfm_D10", "slfm_D25", "slfm_D50", "slfm_D100", "slfm_D500"), "_m", m)

tmp_results <- matrix(nrow = 10, ncol = 9, dimnames = list(c(), c("city.mpg.train", "city.mpg.test", "city.mpg.sigma", "highway.mpg.train", "highway.mpg.test", "highway.mpg.sigma", "price.train", "price.test", "price.sigma")))
for(method in c("sep", "slfm_D5", "slfm_D10", "slfm_D25", "slfm_D50")) results[[method]] <- tmp_results


for(rep in 1:10){
  set.seed(32519+rep)
  train_index <- sample(1:n_all)[1:n_train]
  test_index <- (1:n_all)[!(1:n_all) %in% train_index]
  X_train <- X_all[train_index,]
  X_test <- X_all[test_index,]
  Y_train <- Y_all[train_index,]
  Y_test <- Y_all[test_index,]
  
  sep <- sep_bartFit(Y_train, X_train, X_test, cutpoints, nd = 1000, burn = 500, m = m)
  results[[paste0("sep_m", m)]]
  
  for(D in c(5, 10, 25, 50)){
    tmp_fit <- slfm_bartFit(Y_train, X_train, X_test, cutpoints, nd = 1000, burn = 500, D = D, m = m)
    results[[paste0("slfm_D", D, "_m", m)]][rep, "city.mpg.train"] <- sqrt(mean( (Y_train[,1] - rowMeans(tmp_fit$f_train_samples[,1,]))^2))
    results[[paste0("slfm_D", D, "_m", m)]][rep, "city.mpg.test"] <- sqrt(mean( (Y_test[,1] - rowMeans(tmp_fit$f_test_samples[,1,]))^2))
    results[[paste0("slfm_D", D, "_m", m)]][rep, "city.mpg.sigma"] <- mean(tmp_fit$sigma_samples[1,])
    
    results[[paste0("slfm_D", D, "_m", m)]][rep, "highway.mpg.train"] <- sqrt(mean( (Y_train[,2] - rowMeans(tmp_fit$f_train_samples[,2,]))^2))
    results[[paste0("slfm_D", D, "_m", m)]][rep, "highway.mpg.test"] <- sqrt(mean( (Y_test[,2] - rowMeans(tmp_fit$f_test_samples[,2,]))^2))
    results[[paste0("slfm_D", D, "_m", m)]][rep, "highway.mpg.sigma"] <- mean(tmp_fit$sigma_samples[2,])
    
    results[[paste0("slfm_D", D, "_m", m)]][rep, "price.train"] <- sqrt(mean( (Y_train[,3] - rowMeans(tmp_fit$f_train_samples[,3,]))^2))
    results[[paste0("slfm_D", D, "_m", m)]][rep, "price.test"] <- sqrt(mean( (Y_test[,3] - rowMeans(tmp_fit$f_test_samples[,3,]))^2))
    results[[paste0("slfm_D", D, "_m", m)]][rep, "price.sigma"] <- mean(tmp_fit$sigma_samples[3,])  
  }
  
  slfm_D5 <- slfm_bartFit(Y_train, X_train, X_test, cutpoints, nd = 1000, burn = 500, D = 5, m = 100)
  slfm_D10 <- slfm_bartFit(Y_train, X_train, X_test, cutpoints, nd = 1000, burn = 500, D = 10, m = 100)
  slfm_D25 <- slfm_bartFit(Y_train, X_train, X_test, cutpoints, nd = 1000, burn = 500, D = 25, m = 100)
  slfm_D50 <- slfm_bartFit(Y_train, X_train, X_test, cutpoints, nd = 1000, burn = 500, D = 50, m = 100)
  slfm_D100 <- slfm_bartFit(Y_train, X_train, X_test, cutpoints, nd = 1000, burn = 500, D = 100, m = 100)
 
  for(method in c("sep", "slfm_D5", "slfm_D10", "slfm_D25", "slfm_D50")){
    
    results[[method]][rep, "city.mpg.train"] <- sqrt(mean( (Y_train[,1] - rowMeans(get(method)$f_train_samples[,1,]))^2))
    results[[method]][rep, "city.mpg.test"] <- sqrt(mean( (Y_test[,1] - rowMeans(get(method)$f_test_samples[,1,]))^2))
    results[[method]][rep, "city.mpg.sigma"] <- mean(get(method)$sigma_samples[1,])
    
    results[[method]][rep, "highway.mpg.train"] <- sqrt(mean( (Y_train[,2] - rowMeans(get(method)$f_train_samples[,2,]))^2))
    results[[method]][rep, "highway.mpg.test"] <- sqrt(mean( (Y_test[,2] - rowMeans(get(method)$f_test_samples[,2,]))^2))
    results[[method]][rep, "highway.mpg.sigma"] <- mean(get(method)$sigma_samples[2,])
    
    results[[method]][rep, "price.train"] <- sqrt(mean( (Y_train[,3] - rowMeans(get(method)$f_train_samples[,3,]))^2))
    results[[method]][rep, "price.test"] <- sqrt(mean( (Y_test[,3] - rowMeans(get(method)$f_test_samples[,3,]))^2))
    results[[method]][rep, "price.sigma"] <- mean(get(method)$sigma_samples[3,])
  
  }
}


