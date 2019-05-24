library(Rcpp)
library(RcppArmadillo)
source("scripts/makeCutpoints.R")
source("scripts/predictive_intervals.R")

sourceCpp("src/univariate_BART.cpp")
sourceCpp("src/sep_BART.cpp")
sourceCpp("src/slfm_BART.cpp")

# Load some patient data
raw_data <- read.csv(file = "~/Desktop/323_episode2_timeseries.csv")
colMeans(is.na(raw_data))

keep_cols <- c("Hours", "Diastolic.blood.pressure", "Systolic.blood.pressure", "Mean.blood.pressure",
               "Heart.Rate", "Oxygen.saturation", "Respiratory.rate")
data <- raw_data[,keep_cols]

drop_rows <- which(rowSums(is.na(data)) == ncol(data) - 1) # all observations missing except hour
data <- data[-drop_rows,]


#####
# New on 24 May

X_all <- as.matrix(data[,"Hours"], nrow = nrow(data), ncol = 1)
Y_all <- as.matrix(data[,!colnames(data) == "Hours"], nrow = nrow(data))

X_train <- as.matrix(data[1:100, "Hours"], nrow = 100, ncol = 1)
Y_train <- as.matrix(data[1:100, !colnames(data) == "Hours"], nrow = 100, ncol = ncol(data) - 1)

X_test <- as.matrix(data[101:nrow(data), "Hours"], ncol = 1)
Y_test <- as.matrix(data[101:nrow(data), !colnames(data) == "Hours"], nrow = nrow(data) - 100, ncol = ncol(data) - 1)

cutpoints_train <- makeCutpoints(X_train)
cutpoints_all <- makeCutpoints(X_all)


Y_train_miss <- Y_train
Y_train_miss[25:50, "Mean.blood.pressure"] <- NA
Y_train_miss[10:30, "Respiratory.rate"] <- NA

test_miss <- slfm_BART(Y_train_miss, X_train, X_all, cutpoints_train, D = 10, m = 20, verbose = TRUE)
dimnames(test_miss$f_test_samples) <- list(c(), colnames(Y_train_miss), c())



test_0 <- slfm_BART(Y_all, X_all, X_all, cutpoints_all, D = 10, m = 20, verbose = TRUE)


test_1 <- slfm_BART(Y_train, X_train, X_all, cutpoints_train, D = 10, m = 20, verbose = TRUE)


dimnames(test_0$f_train_samples) <- list(c(), colnames(Y_train),c())
dimnames(test_1$f_train_samples) <- list(c(), colnames(Y_train), c())

dimnames(test_0$f_test_samples) <- list(c(), colnames(Y_train), c())
dimnames(test_1$f_test_samples) <- list(c(), colnames(Y_train), c())


plot(1, xlim = range(X_all), ylim = range(Y_all[,"Mean.blood.pressure"], na.rm = TRUE), type = "n", xlab = "Time", ylab = "MBP")
rect(par("usr")[1], par("usr")[3], X_train[nrow(X_train),1], par("usr")[4], col = rgb(187,187,187,255/6, maxColorValue = 255), border = NA)



points(X_train, Y_train[,"Mean.blood.pressure"], pch = 16, cex = 0.5)
points(X_test, Y_test[,"Mean.blood.pressure"], pch = 1, cex = 0.7)
#lines(X_all, rowMeans(test_0$f_test_samples[,"Mean.blood.pressure",]), col = 'black')
lines(X_train, rowMeans(test_1$f_train_samples[1:nrow(X_train), "Mean.blood.pressure",]), col = 'red', lwd = 1.5)
lines(X_test, rowMeans(test_1$f_test_samples[(nrow(X_train) + 1):(nrow(X_train) + nrow(X_test)), "Mean.blood.pressure",]), col = 'red', lty = 2)
polygon(c(X_all, rev(X_all)), 
        c(apply(test_1$f_test_samples[,"Mean.blood.pressure",], MAR = 1, FUN = quantile, probs = 0.025),
          rev(apply(test_1$f_test_samples[,"Mean.blood.pressure",], MAR = 1, FUN = quantile, probs = 0.975))),
        col = rgb(1, 0, 0, 1/3), border = NA)


lines(X_train, rowMeans(test_miss$f_test_samples[1:nrow(X_train), "Mean.blood.pressure",]), col = 'blue', lwd = 1.5)
lines(X_test, rowMeans(test_miss$f_test_samples[(nrow(X_train) + 1):(nrow(X_train) + nrow(X_test)), "Mean.blood.pressure",]), col = 'blue', lty = 2)
polygon(c(X_all, rev(X_all)),
        c(apply(test_miss$f_test_samples[,"Mean.blood.pressure",], MAR = 1, FUN = quantile, probs = 0.025),
          rev(apply(test_miss$f_test_samples[,"Mean.blood.pressure",], MAR = 1, FUN = quantile, probs = 0.975))),
        col = rgb(0, 0,1,1/3), border = NA)

#abline(h = mean(Y_train[,"Mean.blood.pressure"], na.rm= TRUE), lty = 2, lwd = 0.5)


slfm_D10_m20 <- slfm_BART(Y_train, X_train, X_all, cutpoints, D = 10, m = 20, verbose = TRUE)

lines(X_all, rowMeans(slfm_D10_m20$f_test_samples))

#####


X <- as.matrix(data[,"Hours"], nrow = nrow(data), ncol = 1)
Y <- as.matrix(data[,!colnames(data) == "Hours"], nrow = nrow(data))

cutpoints <- makeCutpoints(X)

uni_test1 <- univariate_BART(Y[,"Respiratory.rate"], X, X, cutpoints, verbose = TRUE)
uni_test2 <- univariate_BART(Y[,"Oxygen.saturation"], X, X, cutpoints, verbose = TRUE)


slfm_D10_m50 <- slfm_BART(Y, X, X, cutpoints, D = 10, m = 50, verbose = TRUE)
slfm_D50_m50 <- slfm_BART(Y, X, X, cutpoints, D = 50, m = 50, verbose = TRUE)
sep_fit <- sep_BART(Y, X, X, cutpoints, verbose = TRUE)

dimnames(slfm_D10_m50$f_test_samples) <- list(c(), colnames(Y), c())
dimnames(slfm_D50_m50$f_test_samples) <- list(c(), colnames(Y), c())
dimnames(sep_fit$f_test_samples) <- list(c(), colnames(Y), c())

#quant_D10_m50 <- bart_quantiles(slfm_D10_m50)
#quant_D50_m50 <- bart_quantiles(slfm_D50_m50)
#quant_sep <- bart_quantiles(sep_fit)


quant_D10_m50 <- f_quantiles(slfm_D10_m50)
quant_D50_m50 <- f_quantiles(slfm_D50_m50)
quant_sep <- f_quantiles(sep_fit)

slfm_D10_m50[["quantiles"]] <- quant_D10_m50
slfm_D50_m50[["quantiles"]] <- quant_D50_m50
sep_fit[["quantiles"]] <- quant_sep

png("images/mimic323.png", height = 4, width = 6, units = "in", res = 300)
par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0), mfrow = c(2,3))
for(x in colnames(Y)){
  y_lim <- range(c(Y[,x], slfm_D10_m50$quantiles[,x,]), slfm_D50_m50$quantiles[,x,], sep_fit$quantiles[,x,], na.rm = TRUE)
  plot(1, type = "n", xlim = range(X), ylim = y_lim, main = x, xlab = "Time", ylab = x, cex.main = 0.8, cex.lab = 0.8, cex.axis = 0.8)
  
  polygon(c(X, rev(X)), c(slfm_D10_m50$quantiles[,x,"0.025"], rev(slfm_D10_m50$quantiles[,x, "0.975"])),col = rgb(0,0,1,1/3), border = NA)
  #polygon(c(X, rev(X)), c(sep_fit$quantiles[,x,"0.025"], rev(sep_fit$quantiles[,x,"0.975"])), col = rgb(1,0,0,1/3), border = NA)
  lines(X, rowMeans(slfm_D10_m50$f_test_samples[,x,]), col = 'blue', lwd = 2)
  #lines(X, rowMeans(sep_fit$f_test_samples[,x,]), col = 'red', lwd = 2)
  
  #lines(X, rowMeans(slfm_D50_m50$f_test_samples[,x,]), col = 'green')
  points(X, Y[,x], pch = 16, cex = 0.5)
  #legend("bottomleft", legend = c("SLFM_BART", "SEP_BART"), col = c("blue", "red"), lty = 1, cex = 0.7, bty = "n")
}
dev.off()
save(slfm_D10_m50, slfm_D50_m50, sep_fit, X, Y, cutpoints, file = "~/Dropbox/Broderick_Group/bart_extensions/data/mimic_example323.RData")


plot(X, Y[,"Oxygen.saturation"],pch = 16, cex = 0.4)
lines(X, rowMeans(uni_test2$f_test_samples), col = 'red')
lines(X, rowMeans(slfm_test$f_test_samples[,"Oxygen.saturation",]), col = 'blue')


plot(X, Y[,"Respiratory.rate"],pch = 16, cex = 0.4)
lines(X, rowMeans(uni_test1$f_test_samples), col = 'red')
lines(X, rowMeans(slfm_test$f_test_samples[,"Respiratory.rate",]), col = 'blue')
