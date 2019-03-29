# Compile toy2 results

q <- 10


method_names <- c("sep", "slfm_D2", "slfm_D5", "slfm_D10", "slfm_D20", "slfm_D50")
full_names <- c()
for(m in c(25, 50, 100, 200)){
  tmp_results <- matrix(nrow = 25, ncol = 3*q + 1, dimnames = list(c(), c(paste0("Task", 1:10, "_train"), paste0("Task", 1:10, "_test"), paste0("sigma", 1:10), "Time")))
  
  results <- list()
  for(method in method_names) results[[paste0(method,"_m", m)]] <- tmp_results
  for(batch in 1:5){
    load(paste0("~/Dropbox/Broderick_Group/bart_extensions/results/toy2_m", m, "_", batch, ".RData"))
    for(method in c("sep", "slfm_D2", "slfm_D5", "slfm_D10", "slfm_D20", "slfm_D50")){
      results[[paste0(method, "_m",m)]][1:5 + 5*(batch-1),] <- get(paste0("results_m",m,"_",batch))[[paste0(method, "_m",m)]][1:5,]
    }
    rm(list = paste0("results_m",m,"_",batch))
  }
  full_names <- c(full_names, paste0(method_names, "_m", m))
  assign(paste0("results_m", m), results)
}

train_error <- data.frame("Task1_train" = rep(NA, times = length(full_names)))
test_error <- data.frame("Task1_train" = rep(NA, times = length(full_names)))
sigma <- data.frame("sigma1" = rep(NA, times = length(full_names)))
time <- data.frame("Time" = rep(NA, times = length(full_names)))
for(task in 2:q){
  train_error[paste0("Task", task, "_train")] <- rep(NA, times = length(full_names))
  test_error[paste0("Task",task,"_test")] <- rep(NA, times = length(full_names))
  sigma[paste0("sigma",task)] <- rep(NA, times = length(full_names))
}
rownames(train_error) <- full_names
rownames(test_error) <- full_names
rownames(sigma) <- full_names
rownames(time) <- full_names

for(m in c(25, 50, 100, 200)){
  for(method in method_names){
    
    tmp_means_train <- colMeans(get(paste0("results_m",m))[[paste0(method,"_m",m)]][,paste0("Task", 1:10, "_train")])
    tmp_sd_train <- apply(get(paste0("results_m",m))[[paste0(method, "_m",m)]][,paste0("Task",1:10,"_train")], FUN = sd, MAR = 2)
    
    tmp_means_test <- colMeans(get(paste0("results_m",m))[[paste0(method, "_m",m)]][,paste0("Task",1:10, "_test")])
    tmp_sd_test <- apply(get(paste0("results_m",m))[[paste0(method, "_m",m)]][,paste0("Task",1:10,"_test")], FUN = sd, MAR = 2)
    
    tmp_means_sigma <- colMeans(get(paste0("results_m",m))[[paste0(method, "_m",m)]][,paste0("sigma",1:10)])
    tmp_sd_sigma <- apply(get(paste0("results_m",m))[[paste0(method,"_m",m)]][,paste0("sigma",1:10)], FUN = sd, MAR = 2)
    
    tmp_means_time <- mean(get(paste0("results_m",m))[[paste0(method, "_m",m)]][,"Time"])
    tmp_sd_time <- sd(get(paste0("results_m",m))[[paste0(method, "_m",m)]][,"Time"])
    
    train_error[paste0(method, "_m", m),] <- paste0(round(tmp_means_train, digits = 3), " (", round(tmp_sd_train, digits = 3), ")")
    test_error[paste0(method, "_m",m),] <- paste0(round(tmp_means_test, digits = 3), " (", round(tmp_sd_test, digits = 3), ")")
    sigma[paste0(method, "_m", m),] <- paste0(round(tmp_means_sigma, digits = 3), " (", round(tmp_sd_sigma, digits = 3), ")")
    time[paste0(method, "_m",m),"Time"] <- paste0(round(tmp_means_time, digits = 3), " (", round(tmp_sd_time, digits = 3), ")")
  }
}

save(train_error, test_error, sigma, time, results_m25, results_m50, results_m100, results_m200, file = "~/Dropbox/Broderick_Group/bart_extensions/results/toy2_results.RData")
