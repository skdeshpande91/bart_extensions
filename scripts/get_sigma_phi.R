prob_func <- function(sigma_phi, y_max, y_min, df =  10, target_prob = 0.9){
  tau <- sqrt(rgamma(1e5, shape = df/2, rate = 1/(2 * sigma_phi * sigma_phi))) # tau^2 ~ sigma_phi^2 * chisq_df
  prob <- mean(pnorm(y_max/tau) - pnorm(y_min/tau))
  #return(prob - target_prob)
  return(target_prob - prob)
}


get_sigma_phi <- function(Y, target_prob = 0.9, df = 10, maxiter = 1000){
  if(mean(Y, na.rm = TRUE) != 0 || sd(Y, na.rm = TRUE) != 1){
    mu_y <- mean(Y, na.rm = TRUE)
    sigma_y <- sd(Y, na.rm = TRUE)
    Y <- (Y - mu_y)/sigma_y
  }
  y_max <- max(Y, na.rm = TRUE)
  y_min <- min(Y, na.rm = TRUE)
 
  tmp <- uniroot(prob_func, interval = c(0, 1e3), y_max = y_max, y_min = y_min, df = df, target_prob = target_prob, maxiter = maxiter)
  return(tmp$root)
}