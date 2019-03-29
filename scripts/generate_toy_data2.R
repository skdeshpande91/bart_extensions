library(MASS)
source("scripts/makeCutpoints.R")

n <- 1000
p <- 2
q <- 10
D <- 4

sigma <- runif(q, 0.25, 0.8)

set.seed(31119)
#X_train <- cbind("X1" = runif(n), "X2" = 1*(runif(n) > 0.5))
X_train <- cbind("X1" = sort(runif(n)), "X2" = runif(n))

X_test <- cbind("X1" = sort(runif(100), decreasing = FALSE), "X2" = runif(100))

#X_test <- cbind("X1" = c(seq(0, 1, by = 0.01), seq(0,1,by = 0.01)), "X2" = c(rep(0, times = 101), rep(1, times = 101)))

cutpoints <- makeCutpoints(X_train)


##########################################
# Generate some true basis functions. These are all extraordinarly simple but illustrative
# u_1 ~ 5 * X1 * X1
# u_2 ~ 3 * X1 + (2 - 5 * (X2 > 0.5)) * sin(PI * X1) - 2 * (X2 > 0.5)
# u_3 ~ GP(SE_1)
# u_4 ~ GP(SE_1 * PER_1)
##########################################
#x1_test <- X_test[X_test[,2] == 0,1]


u1_train <- 5 * X_train[,1] * X_train[,1]
u1_test <- 5 * X_test[,1] * X_test[,1]
u1_all <- c(u1_train, u1_test)

u2_train <- 3 * X_train[,1] + (2 - 5 * (X_train[,2] > 0.5)) * sin(pi * X_train[,1]) - 2 * (X_train[,2] > 0.5)
u2_test <- 3 * X_test[,1] + (2 - 5 * (X_test[,2] > 0.5)) * sin(pi * X_test[,1]) - 2 * (X_test[,2] > 0.5)


# For u3 we need to construct a GP kernel. Use length scale = 0.25 for this squared exponential kernel
x1_all <- c(X_train[,1], X_test[,1])
k3_all <- 1*exp(-1/(0.25* 0.25) * outer(x1_all, x1_all,FUN = "-") * outer(x1_all, x1_all, FUN = "-"))
set.seed(318)
u3_all <- mvrnorm(n = 1, mu = rep(0, times = length(x1_all)), Sigma = k3_all)
u3_train <- u3_all[1:n]
u3_test <- u3_all[(n+1):length(x1_all)]

k4_all <- 1*exp(-1/(2 * 0.1 * 0.1) * outer(x1_all, x1_all, FUN = "-") * outer(x1_all, x1_all, FUN = "-")) * exp(-2/(0.1*0.1) * sin(pi*outer(x1_all, x1_all, FUN = "-")/2) * sin(pi*outer(x1_all, x1_all, FUN = "-")/2))
set.seed(1123)
u4_all <- mvrnorm(n = 1, mu = rep(0, times = length(x1_all)), Sigma = k4_all)
u4_train <- u4_all[1:n]
u4_test <- u4_all[(n+1):length(x1_all)]

###########################################
# Toy Data 2:
set.seed(190318)
Phi <- matrix(rnorm(q*D, mean = 0, sd = 1), nrow = q, ncol = D)
Y <- matrix(nrow = n, ncol = q)
f_max <- 0



save_list <- c("Phi", "Y", "X_test", "X_train", "sigma")
for(d in 1:D){
  save_list <- c(save_list, paste0("u", d, "_train"), paste0("u", d, "_test"))
}

for(k in 1:q){
  set.seed(220319 + k)
  tmp_train <- Phi[k,1] * u1_train + Phi[k,2] * u2_train + Phi[k,3] * u3_train + Phi[k,4] * u4_train
  tmp_test <- Phi[k,1] * u1_test + Phi[k,2] * u2_test + Phi[k,3] * u3_test + Phi[k,4] * u4_test
  if(max(abs(c(tmp_train, tmp_test))) > f_max) f_max <- max(abs(c(tmp_train, tmp_test)))
  
  Y[,k] <- tmp_train + sigma[k]*rnorm(n, 0, 1)
  
  assign(paste0("f", k, "_train"), tmp_train)
  assign(paste0("f", k, "_test"), tmp_test)
  
  save_list <- c(save_list, paste0("f",k, "_train"), paste0("f", k, "_test"))
}

save(list = save_list, file = "data/toy_example2.RData")
