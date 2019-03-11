source("scripts/makeCutpoints.R")

n <- 1000
p <- 2
q <- 2
D <- 2

sigma_1 <- 0.75
sigma_2 <- 0.5

set.seed(31119)
X_train <- cbind("X1" = runif(n), "X2" = runif(n))
cutpoints <- makeCutpoints(X_train)

# Generate the training data
x1 <- X_train[,"X1"]
x2 <- X_train[,"X2"]
x2_tmp <- 1*(x2 > 0.5)
max_tmp <- x1 * (x1 >= x2) + x2 *(x2 >= x1)

#u1_train <- 5 * x1 * x1 - 3 * max_tmp
u1_train <- 5 * x1 * x1
u2_train <- 3 * x1 + (2 - 5*x2_tmp) * sin(pi * x1) - 2 * x2_tmp

Phi <- matrix(rnorm(q*D, mean = 0, sd = 5), nrow = q, ncol = D)

f1_train <- Phi[1,1] * u1_train + Phi[1,2] * u2_train
f2_train <- Phi[2,1] * u1_train + Phi[2,2] * u2_train

y1 <- f1_train + sigma_1 * rnorm(n, 0, 1)
y2 <- f2_train + sigma_2 * rnorm(n, 0, 1)

Y <- cbind("Y1" = y1, "Y2" = y2)


# Plot the basis functions, true functions with and without the data
x_seq <- seq(0, 1, by = 0.01)
u1_full <- 5 * x_seq * x_seq
u2_full_0 <- 3 * x_seq + (2 - 5*0)*sin(pi * x_seq) - 2 * 0
u2_full_1 <- 3 * x_seq + (2 - 5*1) * sin(pi * x_seq) - 2*1

f1_full_0 <- Phi[1,1] * u1_full + Phi[1,2] * u2_full_0
f1_full_1 <- Phi[1,1] * u1_full + Phi[1,2] * u2_full_1
f2_full_0 <- Phi[2,1] * u1_full + Phi[2,2] * u2_full_0
f2_full_1 <- Phi[2,1] * u1_full + Phi[2,2] * u2_full_1

f_range <- c(-1,1) * max(abs(c(f1_full_0, f1_full_1, f2_full_0, f2_full_1)))
u_range <- c(-1,1) * max(abs(c(u1_full, u2_full_0, u2_full_1)))


# Before plotting, save everything for later use
save(X_train, u1_train, u2_train, Phi, f1_train, f2_train, Y, sigma_1, sigma_2, u1_full, u2_full_0, u2_full_1, f1_full_0, f1_full_1, f2_full_0, f2_full_1, cutpoints, 
     file = "data/toy_example.RData")

png("images/toy_example_no_data.png", width = 6, height = 6, units = "in", res = 300)
par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0), mfrow = c(2,2), cex = 0.8, cex.lab = 0.8, cex.axis = 0.8, cex.main = 0.9)
plot(1, type = "n", xlab = expression(X[1]), ylab = expression(u[1]), main = expression(Basis ~ Function ~ u[1]), xlim = c(0,1), ylim = u_range)
lines(x_seq, u1_full, col = 'red', lwd = 2)

plot(1, type = "n", xlab = expression(X[1]),ylab = expression(u[1]), main = expression(Basis ~ Function ~ u[2]), xlim = c(0,1), ylim = u_range)
lines(x_seq, u2_full_0, col = 'red', lty = 1, lwd = 2)
lines(x_seq, u2_full_1, col = 'red', lty = 2, lwd = 2)
legend("topleft", legend = c(expression(x[2]<=0.5), expression(x[2]>0.5)), lty = c(1,2), bty = "n", col = 'red',cex = 0.8)

plot(1, type = "n", xlab= expression(X[1]), ylab = expression(f[1]), main = expression(Task ~ f[1]), xlim = c(0,1), ylim = f_range)
lines(x_seq, f1_full_0, col = 'red', lty = 1, lwd = 2)
lines(x_seq, f1_full_1, col = 'red', lty = 2, lwd = 2)
#legend(x = 0, y = 10, legend = c(expression(x[2] <= 0.5), expression(x[2] > 0.5)), lty = c(1,2), bty = "n", col = 'red')
legend("topleft", legend = c(expression(x[2] <= 0.5), expression(x[2] > 0.5)), lty = c(1,2), bty = "n", col = 'red', cex = 0.8)
legend("bottomleft", legend = c(paste("Phi[1,1] =", round(Phi[1,1], digits = 3)), paste("Phi[1,2] =", round(Phi[1,2], digits = 3))), 
       bty = "n", cex = 0.8)


#points(x1, y1, pch = 16, cex = 0.5)
plot(1, type = "n", xlab = expression(X[1]), ylab = expression(f[2]), main = expression(Task ~ f[2]), xlim = c(0,1), ylim = f_range)
lines(x_seq, f2_full_0, col = 'red', lty = 1, lwd = 2)
lines(x_seq, f2_full_1, col = 'red', lty = 2, lwd = 2)
#legend(x = 0, y = 10, legend = c(expression(x[2] <= 0.5), expression(x[2] > 0.5)), lty = c(1,2), bty = "n", col = 'red')
legend("topleft", legend = c(expression(x[2] <= 0.5), expression(x[2] > 0.5)), lty = c(1,2), bty = "n", col = 'red', cex = 0.8)
legend("bottomleft", legend = c(paste("Phi[2,1] =", round(Phi[2,1], digits = 3)), paste("Phi[2,2] =", round(Phi[1,2], digits = 3))), 
       bty = "n", cex = 0.8)
#points(x1, y2, pch = 16, cex = 0.5)

dev.off()

png("images/toy_example_data.png", width = 6, height = 6, units = "in", res = 300)
par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0), mfrow = c(2,2), cex = 0.8, cex.lab = 0.8, cex.axis = 0.8, cex.main = 0.9)
plot(1, type = "n", xlab = expression(X[1]), ylab = expression(u[1]), main = expression(Basis ~ Function ~ u[1]), xlim = c(0,1), ylim = u_range)
lines(x_seq, u1_full, col = 'red', lwd = 2)


plot(1, type = "n", xlab = expression(X[1]),ylab = expression(u[1]), main = expression(Basis ~ Function ~ u[2]), xlim = c(0,1), ylim = u_range)
lines(x_seq, u2_full_0, col = 'red', lty = 1, lwd = 2)
lines(x_seq, u2_full_1, col = 'red', lty = 2, lwd = 2)
legend("topleft", legend = c(expression(x[2]<=0.5), expression(x[2]>0.5)), lty = c(1,2), bty = "n", col = 'red',cex = 0.8)

plot(1, type = "n", xlab= expression(X[1]), ylab = expression(f[1]), main = expression(Task ~ f[1]), xlim = c(0,1), ylim = f_range)
points(x1, y1, pch = 16, cex = 0.3)

lines(x_seq, f1_full_0, col = 'red', lty = 1, lwd = 2)
lines(x_seq, f1_full_1, col = 'red', lty = 2, lwd = 2)
#legend(x = 0, y = 10, legend = c(expression(x[2] <= 0.5), expression(x[2] > 0.5)), lty = c(1,2), bty = "n", col = 'red')
legend("topleft", legend = c(expression(x[2] <= 0.5), expression(x[2] > 0.5)), lty = c(1,2), bty = "n", col = 'red', cex = 0.8)
legend("bottomleft", legend = c(paste("Phi[1,1] =", round(Phi[1,1], digits = 3)), paste("Phi[1,2] =", round(Phi[1,2], digits = 3))), 
       bty = "n", cex = 0.8)



plot(1, type = "n", xlab = expression(X[1]), ylab = expression(f[2]), main = expression(Task ~ f[2]), xlim = c(0,1), ylim = f_range)
points(x1, y2, pch = 16, cex = 0.3)
lines(x_seq, f2_full_0, col = 'red', lty = 1, lwd = 2)
lines(x_seq, f2_full_1, col = 'red', lty = 2, lwd = 2)
#legend(x = 0, y = 10, legend = c(expression(x[2] <= 0.5), expression(x[2] > 0.5)), lty = c(1,2), bty = "n", col = 'red')
legend("topleft", legend = c(expression(x[2] <= 0.5), expression(x[2] > 0.5)), lty = c(1,2), bty = "n", col = 'red', cex = 0.8)
legend("bottomleft", legend = c(paste("Phi[2,1] =", round(Phi[2,1], digits = 3)), paste("Phi[2,2] =", round(Phi[2,2], digits = 3))), 
       bty = "n", cex = 0.8)


dev.off()



