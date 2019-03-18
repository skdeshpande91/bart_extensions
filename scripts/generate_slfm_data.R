library(MASS)
source("scripts/makeCutpoints.R")

n <- 1000
p <- 2
q <- 2
D <- 4

sigma_1 <- 0.75
sigma_2 <- 0.5

set.seed(31119)
X_train <- cbind("X1" = runif(n), "X2" = runif(n))

X_test <- cbind("X1" = c(seq(0, 1, by = 0.01), seq(0,1,by = 0.01)), "X2" = c(rep(0, times = 101), rep(1, times = 101)))

cutpoints <- makeCutpoints(X_train)


##########################################
# Generate some true basis functions. These are all extraordinarly simple but illustrative
# u_1 ~ 5 * X1 * X1
# u_2 ~ 3 * X1 + (2 - 5 * (X2 > 0.5)) * sin(PI * X1) - 2 * (X2 > 0.5)
# u_3 ~ GP(SE_1)
# u_4 ~ GP(SE_1 * PER_1)
##########################################
x1_test <- X_test[X_test[,2] == 0,1]

u1_train <- 5 * X_train[,1] * X_train[,1]
u1_test <- 5 * x1_test * x1_test
u1_all <- c(u1_train, u1_test)


u2_train <- 3 * X_train[,1] + (2 - 5 * (X_train[,2] > 0.5)) * sin(pi * X_train[,1]) - 2*(X_train[,2] > 0.5)
u2_test_0 <- 3 * x1_test + (2 - 5*0) * sin(pi * x1_test) - 2*0
u2_test_1 <- 3 * x1_test + (2 - 5*1) * sin(pi * x1_test) - 2*1
u2_all <- c(u2_train, u2_test_0, u2_test_1)

# For u3 we need to construct a GP kernel. Use length scale = 0.01 for this squared exponential kernel
x1_all <- c(X_train[,1], x1_test)
k3_all <- 1*exp(-1/(0.25* 0.25) * outer(x1_all, x1_all,FUN = "-") * outer(x1_all, x1_all, FUN = "-"))
set.seed(318)
u3_all <- mvrnorm(n = 1, mu = rep(0, times = length(x1_all)), Sigma = k3_all)
u3_train <- u3_all[1:n]
u3_test <- u3_all[(n+1):length(x1_all)]

k4_all <- 1*exp(-1/(2 * 0.1 * 0.1) * outer(x1_all, x1_all, FUN = "-") * outer(x1_all, x1_all, FUN = "-")) * exp(-2/(0.1*0.1) * sin(pi*outer(x1_all, x1_all, FUN = "-")/2) * sin(pi*outer(x1_all, x1_all, FUN = "-")/2))
set.seed(1123)
u4_all <- mvrnorm(n = 1, mu = rep(0, times = length(x1_all)), Sigma = k4_all)
u4_train <- u4_all[1:n]
u4_test <- u4_all[(n+1):(n+length(x1_test))]

###########################################
# Toy Data:
set.seed(190318)
Phi <- matrix(rnorm(q*D, mean = 0, sd = 1), nrow = q, ncol = D)

f1_train <- Phi[1,1] * u1_train + Phi[1,2] * u2_train + Phi[1,3] * u3_train + Phi[1,4] * u4_train
f1_test_0 <- Phi[1,1] * u1_test + Phi[1,2]*u2_test_0 + Phi[1,3]*u3_test + Phi[1,4] * u4_test
f1_test_1 <- Phi[1,1] * u1_test + Phi[1,2]*u2_test_1 + Phi[1,3]*u3_test + Phi[1,4]* u4_test

f2_train <- Phi[2,1] * u1_train + Phi[2,2] * u2_train + Phi[2,3] * u3_train + Phi[2,4] * u4_train
f2_test_0 <- Phi[2,1] * u1_test + Phi[2,2] * u2_test_0 + Phi[2,3] * u3_test + Phi[2,4] * u4_test
f2_test_1 <- Phi[2,1] * u1_test + Phi[2,2] * u2_test_1 + Phi[2,3] * u3_test + Phi[2,4] * u4_test

u_range <- c(-1,1) * max(abs(c(u1_all, u2_all, u3_all, u4_all)))
f_range <- c(-1,1) * max(abs(c(f1_test_0, f1_test_1, f2_test_0, f2_test_1, f1_train, f2_train)))


y1 <- f1_train + sigma_1*rnorm(n,0,1)
y2 <- f2_train + sigma_2*rnorm(n,0,1)
Y <- cbind("Y1" = y1, "Y2" = y2)

# Plot the basis functions

png("images/toy_example_basis.png", width = 8, height = 2, units = "in", res = 300)
par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0), mfrow = c(1,4), cex.main = 0.9, cex.lab = 0.8, cex.axis = 0.8)
plot(1, type = "n", xlab = expression(X[1]), ylab = expression(u[1]), main = expression(Basis ~ Function ~ u[1]), xlim = c(0,1), ylim = u_range)
lines(x1_test, u1_test)

plot(1, type = "n", xlab = expression(X[1]), ylab = expression(u[2]), main = expression(Basis ~ Function ~ u[2]), xlim = c(0,1), ylim = u_range)
lines(x1_test, u2_test_0, lty = 1)
lines(x1_test, u2_test_1, lty = 2)
legend("topleft", legend = c(expression(x[2]<=0.5), expression(x[2]>0.5)), lty = c(1,2), bty = "n", col = 'black',cex = 0.8)

plot(1, type = "n", xlab = expression(X[1]), ylab = expression(u[3]), main = expression(Basis ~ Function ~ u[3]), xlim = c(0,1), ylim = u_range)
lines(x1_test, u3_test)

plot(1, type = "n", xlab = expression(X[1]), ylab = expression(u[4]), main = expression(Basis ~ Function ~ u[4]), xlim = c(0,1), ylim = u_range)
lines(x1_test, u4_test)
dev.off()

png("images/toy_example_no_data.png", width = 6, height = 6, units = "in", res = 300)
par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0), mfrow = c(1,1), cex.main = 0.9, cex.lab = 0.8, cex.axis = 0.8)
plot(1, type = "n", xlab = expression(X[1]), ylab = expression(f), main = "Tasks 1 & 2", xlim = c(0,1), ylim = f_range)

lines(x1_test, f1_test_0, col = 'red', lty = 1)
lines(x1_test, f1_test_1, col = 'red', lty = 2)
#points(X_train[,1], Y[,1], pch = 3, cex = 0.5)

lines(x1_test, f2_test_0, col = 'blue', lty = 1)
lines(x1_test, f2_test_1, col = 'blue', lty = 2)
#points(X_train[,1], Y[,2], pch = 4, cex = 0.5)


#legend("topleft", legend = c(expression(f[1]), expression(f[2])), pch = c(3,4), text.col = c("red", "blue"), bty = "n", cex = 0.8)
legend("topleft", legend = c(expression(f[1]), expression(f[2])), text.col = c('red', 'blue'), bty = 'n', cex = 0.8)
legend("bottomleft", legend = c(expression(x[2] <= 0.5), expression(x[2] > 0.5)), lty = c(1,2), bty = "n", col = 'black', cex = 0.8)
dev.off()

png("images/toy_example_data.png", width = 6, height = 6, units = "in", res = 300)
par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0), mfrow = c(1,1), cex.main = 0.9, cex.lab = 0.8, cex.axis = 0.8)
plot(1, type = "n", xlab = expression(X[1]), ylab = expression(f), main = "Tasks 1 & 2", xlim = c(0,1), ylim = f_range)

lines(x1_test, f1_test_0, col = 'red', lty = 1)
lines(x1_test, f1_test_1, col = 'red', lty = 2)
points(X_train[,1], Y[,1], pch = 3, cex = 0.5)

lines(x1_test, f2_test_0, col = 'blue', lty = 1)
lines(x1_test, f2_test_1, col = 'blue', lty = 2)
points(X_train[,1], Y[,2], pch = 4, cex = 0.5)


legend("topleft", legend = c(expression(f[1]), expression(f[2])), pch = c(3,4), text.col = c("red", "blue"), bty = "n", cex = 0.8)
#legend("topleft", legend = c(expression(f[1]), expression(f[2])), text.col = c('red', 'blue'), bty = 'n', cex = 0.8)
legend("bottomleft", legend = c(expression(x[2] <= 0.5), expression(x[2] > 0.5)), lty = c(1,2), bty = "n", col = 'black', cex = 0.8)
dev.off()


save(X_train, X_test, Y, Phi,
     u1_train, u1_test, u2_train, u2_test_0, u2_test_1, u3_train, u3_test, u4_train, u4_test, 
     f1_train, f2_train, f1_test_0, f1_test_1, f2_test_0, f2_test_1,
     sigma_1, sigma_2, f_range, u_range, file = "data/toy_example.RData")