# Make cutpoints
makeCutpoints <- function(X, gridlen = 10000){
  cutpoints <- list(); # empty list which will contain cutpoints
  for(j in 1:ncol(X)){
    
    # Check if X[,j] is a binary indicator
    if(all(X[,j] %in% c(0,1))){
      cutpoints[[j]] <- c(0,1)
    } else{
      min_X <- min(X[,j])
      max_X <- max(X[,j])
      cutpoints[[j]] <- seq(min_X, max_X, length.out = gridlen) # why aren't we just using the observed values of X[,j] as cutpoints?
    }
  }
  return(cutpoints)
}
