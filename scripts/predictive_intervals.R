
#load("~/Dropbox/Broderick_Group/bart_extensions/data/airtemp.RData")
#load("~/Dropbox/Broderick_Group/bart_extensions/airtemp_example.RData")

#fit <- slfm_D5_m50

pbart_pred <- function(t, prob, i, k, fit){
  # i is the index of observation
  # k is the task
  # We want P(y*_ik < t | y)
  return(prob - mean(pnorm( (t - fit$f_test_samples[i,k,])/fit$sigma_samples[k,], mean  = 0, sd = 1)))
}


get_pred_quantiles <- function(fit, prob = 0.95, i, k){
  
  
  root_range <- range(fit$f_test_samples[i,k,] + 5 * fit$sigma_samples[k,], fit$f_test_samples[i,k,] - 5 * fit$sigma_samples[k,])
  tmp <- tryCatch(uniroot(f = pbart_pred, interval = root_range, i = i, k = k, prob = prob, fit = fit),
                  error = function(e){NULL}, warning = function(e){NULL})
  if(!is.null(tmp)){
    return(tmp$root)
  } else{
    return(NULL)
  }
}


bart_quantiles <- function(fit, prob = c(0.025, 0.975)){
  
  n_pred <- dim(fit$f_test_samples)[1]
  q <- dim(fit$f_test_samples)[2]
  
  quantiles <- array(NA,dim = c(n_pred, q, length(prob)), dimnames = list(c(), colnames(fit$f_test_samples), prob))
  for(i in 1:n_pred){
    for(k in 1:q){
      for(pr in prob){
        quantiles[i,k,as.character(pr)] <- get_pred_quantiles(fit, prob = pr, i = i, k = k)
      }
    }
  }
  return(quantiles)
}

f_quantiles <- function(fit, prob = c(0.025, 0.975)){
  n_pred <- dim(fit$f_test_samples)[1]
  q <- dim(fit$f_test_samples)[2]
  quantiles <- array(NA, dim = c(n_pred, q, length(prob)), dimnames = list(c(), colnames(fit$f_test_samples), prob))
  for(i in 1:n_pred){
    for(k in 1:q){
      for(pr in prob){
        quantiles[i,k,as.character(pr)] <- quantile(fit$f_test_samples[i,k,], probs = pr)
      }
    }
  }
  return(quantiles)
}
