library(Rcpp)
library(RcppArmadillo)

source("scripts/makeCutpoints.R")
load("data/toy_example.RData")


cutpoints <- makeCutpoints(X_train, gridlen = 10000)

sourceCpp("src/slfm_bart_sparse.cpp")
sourceCpp("src/slfm_bart.cpp")

dense_test <- slfm_bartFit(Y, X_train, X_test, cutpoints, nd = 1000, burn = 500, D = 20, m = 100)

sparse_test1 <- slfm_bartFit_sparse(Y, X_train, X_test, cutpoints, a_theta = 1, b_theta = 1, nd = 1000, burn = 500, D = 20, m = 100)
sparse_test2 <- slfm_bartFit_sparse(Y, X_train, X_test, cutpoints, a_theta = 1, b_theta = 10, nd = 1000, burn = 500, D = 20, m = 100)
sparse_test3 <- slfm_bartFit_sparse(Y, X_train, X_test, cutpoints, a_theta = 10, b_theta = 1, nd = 1000, burn = 500, D = 20, m = 100)
sparse_test4 <- slfm_bartFit_sparse(Y, X_train, X_test, cutpoints, a_theta = 10, b_theta = 10, nd = 1000, burn = 500, D = 20, m = 100)
sparse_test5 <- slfm_bartFit_sparse(Y, X_train, X_test, cutpoints, a_theta = 100, b_theta = 100, nd = 1000, burn= 500, D = 20, m = 100)
sparse_test6 <- slfm_bartFit_sparse(Y, X_train, X_test, cutpoints, a_theta = 100, b_theta = 1, nd = 1000, burn= 500, D = 20, m = 100)
sparse_test7 <- slfm_bartFit_sparse(Y, X_train, X_test, cutpoints, a_theta = 1, b_theta = 100, nd = 1000, burn= 500, D = 20, m = 100)



plot(X_train[,1], rowMeans(dense_test$f_train_samples[,1,]), pch = 16, cex = 0.6, col = 'black')
points(X_train[,1], rowMeans(sparse_test1$f_train_samples[,1,]), pch = 16, cex = 0.5, col = 'blue')
points(X_train[,1], rowMeans(sparse_test2$f_train_samples[,1,]), pch = 3, cex = 0.5, col = 'green')
points(X_train[,1], rowMeans(sparse_test3$f_train_samples[,1,]), pch = 4, cex = 0.5, col = 'yellow')
points(X_train[,1], rowMeans(sparse_test4$f_train_samples[,1,]), pch = 5, cex = 0.5, col = 'purple')


points(X_train[,1], rowMeans(sparse_test5$f_train_samples[,1,]), pch = 3, cex = 0.5, col = 'blue')
points(X_train[,1], rowMeans(sparse_test6$f_train_samples[,1,]), pch = 4, cex = 0.5, col = 'red')
points(X_train[,1], rowMeans(sparse_test7$f_train_samples[,1,]), pch = 5, cex = 0.5, col = 'green') # really demonstrates that we have agressive shrinkage to the mean
abline(h = mean(Y[,1]), col = 'red')
abline(h = mean(Y[,1]), col = 'red')


sparse_test1 <- slfm_bartFit_sparse(Y, X_train, X_test, cutpoints, a_theta = 1, b_theta = 1, nd = 1000, burn = 500, D = 10, m = 200)
sparse_test2 <- slfm_bartFit_sparse(Y, X_train, X_test, cutpoints, a_theta = 10, b_theta = 10, nd = 1000, burn = 500, D = 10, m = 200)
sparse_test3 <- slfm_bartFit_sparse(Y, X_train, X_test, cutpoints, a_theta = 1000, b_theta = 1000, nd = 1000, burn = 500, D = 10, m = 200)
sparse_test4 <- slfm_bartFit_sparse(Y, X_train, X_test, cutpoints, a_theta = 100, b_theta = 1, nd = 1000, burn = 500, D = 10, m = 200)






# How many basis functions used for task 1
rowMeans(dense_test$Phi[1,,] != 0) 
rowMeans(sparse_test1$Phi_samples[1,,] != 0) # just about 10% --> only about 1 tree
rowMeans(sparse_test1$theta_samples) # theta concentrates around 0.09
# Is it possible that 


rowMeans(sparse_test2$Phi_samples[1,,] != 0) # just about 35% --> only about 3 trees
rowMeans(sparse_test2$theta_samples) # theta concentrates around 0.34

rowMeans(sparse_test3$Phi_samples[1,,] != 0) # around 49%


plot(X_train[,1], rowMeans(sparse_test1$f_train_samples[,1,]), ylim = f_range, pch = 16, cex = 0.)
abline(h = mean(Y[,1]), col = 'red')


sparse_test3 <- slfm_bartFit_sparse(Y, X_train, X_test, cutpoints, a_theta = 100, b_theta = 1, nd = 1000, burn = 500, D = 10, m = 200)
sparse_test4 <- slfm_bartFit_sparse(Y, X_train, X_test, cutpoints, a_theta = 1000, b_theta = 0.001, nd = 1000, burn = 500, D = 10, m = 200)
sparse_test5 <- slfm_bartFit_sparse(Y, X_train, X_test, cutpoints, a_theta = 10, b_theta = 10, nd = 1000, burn = 500, D = 10, m = 200)
sparse_test6 <- slfm_bartFit_sparse(Y, X_train, X_test, cutpoints, a_theta = 100, b_theta = 100, nd = 1000, burn = 500, D = 10, m = 200)



sqrt(mean( (c(f1_test_0, f1_test_1) - rowMeans(dense_test$f_test_samples[,1,]))^2))
sqrt(mean( (c(f2_test_0, f2_test_1) - rowMeans(dense_test$f_test_samples[,2,]))^2))


sqrt(mean( (c(f1_test_0, f1_test_1) - rowMeans(sparse_test1$f_test_samples[,1,]))^2))
sqrt(mean( (c(f2_test_0, f2_test_1) - rowMeans(sparse_test1$f_test_samples[,2,]))^2))

sqrt(mean( (c(f1_test_0, f1_test_1) - rowMeans(sparse_test2$f_test_samples[,1,]))^2))
sqrt(mean( (c(f2_test_0, f2_test_1) - rowMeans(sparse_test2$f_test_samples[,2,]))^2))


sqrt(mean( (c(f1_test_0, f1_test_1) - rowMeans(sparse_test3$f_test_samples[,1,]))^2))
sqrt(mean( (c(f2_test_0, f2_test_1) - rowMeans(sparse_test3$f_test_samples[,2,]))^2))

sqrt(mean( (c(f1_test_0, f1_test_1) - rowMeans(sparse_test4$f_test_samples[,1,]))^2))
sqrt(mean( (c(f2_test_0, f2_test_1) - rowMeans(sparse_test4$f_test_samples[,2,]))^2))

sqrt(mean( (c(f1_test_0, f1_test_1) - rowMeans(sparse_test5$f_test_samples[,1,]))^2))
sqrt(mean( (c(f2_test_0, f2_test_1) - rowMeans(sparse_test5$f_test_samples[,2,]))^2))

sqrt(mean( (c(f1_test_0, f1_test_1) - rowMeans(sparse_test6$f_test_samples[,1,]))^2))
sqrt(mean( (c(f2_test_0, f2_test_1) - rowMeans(sparse_test6$f_test_samples[,2,]))^2))


rowMeans(sparse_test1$Phi_samples[1,,] != 0) # how many basis functions used for task 1
rowMeans(sparse_test2$Phi_samples[1,,] != 0) # how many basis functions used for task 1
rowMeans(sparse_test3$Phi_samples[1,,] != 0) # how many basis functions used for task 1
rowMeans(sparse_test4$Phi_samples[1,,] != 0) # how many basis functions used for task 1
rowMeans(sparse_test5$Phi_samples[1,,] != 0) # how many basis functions used for task 1
rowMeans(sparse_test6$Phi_samples[1,,] != 0) # how many basis functions used for task 1


par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0))
plot(1, type = "n", xlab = expression(X[1]), ylab = "f", main = "Posterior Mean", xlim = c(0,1), ylim = f_range)

points(X_test[,1], rowMeans(sparse_test$f_test_samples[,1,]), pch = 16, cex = 0.5, col = 'red')
points(X_test[,1], rowMeans(sparse_test$f_test_samples[,2,]), pch = 4, cex = 0.5, col = 'blue')

lines(X_test[1:101,1], f1_test_0, col = 'red', lty = 1, lwd = 2)
lines(X_test[102:202,1], f1_test_1, col = 'red', lty = 2, lwd = 2)

lines(X_test[1:101,1], f2_test_0, col = 'blue', lty = 1, lwd = 2)
lines(X_test[102:202,1], f2_test_1, col = 'blue', lty = 2, lwd = 2)

lines(X_test[1:101,1], rowMeans(sparse_test$u_test_samples[1:101,1,]))
plot(X_train[,1], rowMeans(sparse_test$u_train_samples[,5,]))
