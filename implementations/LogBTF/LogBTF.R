## LogBTF (2023), Zhi-Ping Liu all rights reserved
##This program package is supported by the copyright owners and coders "as is" and without warranty of any kind, express or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. In no event shall the copyright owner or contributor be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, without limitation, procurement of substitute goods or services; loss of use, data, or profits; or business interruption), regardless of the theory of liability, whether in contract, strict liability or tort (including negligence or otherwise) for any use of the software, even if advised of the possibility of such damages.





# LogBTFmainfunction ------------------------------------------------------
## 2022.11.28 LogBTF function 

logBTFfunction <- function(method, nfold, numGENES, X_matrix, DISTANCE_matrix){
  # def method
  if(method==1){
    alphas <- 0
  }else if(method==2){
    alphas <- 0.5  
  }else if(method==3){
    alphas <- 1
  }else if(method==4){
    alphas <- seq(0,1,0.1)
  }else{
    input <- readline(' *** Please input manually the alpha values (between 0 and 1) separated by comma: ')
    alphas <- as.numeric(unlist(strsplit(input,',')))
  }
  
  ## LOOCV settings
  foldid <- 1:nfold
  keep <- TRUE
  pred_lambda_min <- matrix(0, nrow = numGENES+1, ncol = numGENES)
  lambda_res <- vector()
  alpha_res <- vector()
  ptrainall0 <- c()
  
  library(glmnet)
  library(pROC)
  for (gi in 1:numGENES) {
    
    # gi <- 3
    ptrainall <- c()
    cverrorall <- c()
    
    lambda <-  vector()
    cvERROR <-  vector()
    beta <- matrix(data=0,nrow = dim(X_matrix)[2],ncol = length(alphas))
    theta <- matrix(data=0,nrow = dim(X_matrix)[2]+1,ncol = length(alphas))
    
    
    # for (test in 1:length(alphas)) {
    test <- 1
    Y_vector <- DISTANCE_matrix[2:(num_time_points),gi]
    
    # if Y exist one 1/0, use noise 0/1 data.
    if(sum(Y_vector) == 0 | sum(Y_vector) == n){
      glm.fit <- glm(Y_vector~., xglm, family = "binomial", control = list(maxit = 100))
      coef <- glm.fit$coefficients
      coef[is.na(coef)] <- 0
      pred_lambda_min[,gi] <- coef
      
    }else if(sum(Y_vector) == 1 | sum(Y_vector) == (n-1)){
      
      glm.fit <- glm(Y_vector~., xglm, family = "binomial", control = list(maxit = 100))
      coef <- glm.fit$coefficients
      coef[is.na(coef)] <- 0
      pred_lambda_min[,gi] <- coef 
      
    }else{
      ## glmnet
      if(noDIAG==1){
        CV_results <- cv.glmnet(X_matrix,Y_vector,alpha=alphas[test],exclude=gi,
                                nfolds = nfold, foldid = foldid, keep = keep, grouped = FALSE)
      }else{
        CV_results <- cv.glmnet(X_matrix,Y_vector,alpha=alphas[test],
                                nfolds = nfold, foldid = foldid, keep = keep, grouped = FALSE)
      }
      
      lambda[test] <- CV_results$lambda.min
      cvERROR[test] <- CV_results$cvm[CV_results$lambda==CV_results$lambda.min]
      cverrorall <- cbind(cverrorall, cvERROR)
      coef.CV_results <- coef(CV_results, s='lambda.min')
      beta[coef.CV_results@i[-1],test] = coef.CV_results@x[-1]
      theta[coef.CV_results@i+1,test] = coef.CV_results@x
      
      
      theta[1,test] <- lambda*theta[1,test]    
      
      minIdx <- max(which(cvERROR==min(cvERROR)))
      lambda_res[gi] <- lambda[minIdx]
      alpha_res[gi] <- alphas[minIdx]
      pred_lambda_min[,gi] <- theta[,minIdx]
      
    }
	
	print(gi) 
       
  }
  return(pred_lambda_min)  
}  


## 2022.7.24
## sign function
sgn <- function(xx){
  y <- xx
  for (k in 1:dim(xx)[1]) {
    # k <- 1
    if (xx[k,1] >= 0)
      y[k,] <- 1
    else (y[k,] <- 0)
  }
  return(y) 
} 


## 2022.11.28 
## Add perturbation design matrix
PerMatrix <- function(datahatA){
  ## noise matrix
  set.seed(2022)
  res <- datahatA[1:(n-1),]
  noiseLevel = 1e-5
  res <- res + matrix(rnorm(mean=0, sd=noiseLevel, n = length(res)), nrow=nrow(res))
  
  ## when 1, it add noise
  for (i in 1:(dim(res)[1])) {
    for (j in 1:dim(res)[2]) {
      if (datahatA[i,j] == 0)
        res[i,j] <- 0
    }
  }
  return(res)
} 