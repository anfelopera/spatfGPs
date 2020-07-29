# model: object containing the trained GP metamodel
# finputs: an 8-length list containing hydro-meteorological time-series [1x37 matrices] 
#  - names(finputs): c("slr","Tide","Surge","Tp","Hsx","Hsy","Ux","Uy")
# x: nx2 matrix containing the spatial location of predictions.
# return_variance: if "return_variance == 1", predicted variances are returned.
predictGPModelMaps <- function(model, finputs_test, x, return_variance = FALSE) {
  ## loading R packages
  require(kergp)

  ## loading local R scripts
  source("rscripts/basisCompute.R")
  source("rscripts/futils.R")
  source("rscripts/corkernelfi.R")
  source("rscripts/kernelsi.R")
  
  ## changing nautical to cartesian coordinates
  finputs_test2 <- finputs_test
  finputs_test2[["Hsx"]] <- finputs_test[["Hs"]] * sin(finputs_test[["Dp"]] * pi/180)
  finputs_test2[["Hsy"]] <- finputs_test[["Hs"]] * cos(finputs_test[["Dp"]] * pi/180)  
  finputs_test2[["Hs"]] <- NULL; finputs_test2[["Dp"]] <- NULL
  finputs_test2[["Ux"]] <- finputs_test[["U"]] * sin(finputs_test[["Du"]] * pi/180)
  finputs_test2[["Uy"]] <- finputs_test[["U"]] * cos(finputs_test[["Du"]] * pi/180)
  finputs_test2[["U"]] <- NULL; finputs_test2[["Du"]] <- NULL
  finputs_test <- finputs_test2
  
  ## defining some parameters
  nbspat <- c(530, 790) # nb points per spatial dimension
  finputs_train <- model$covariance@covAlls$kfunc@groupList # training set of functional inputs
  nrep <- length(finputs_train$coef) # nb training maps
  nfinputs <- length(finputs_train$f) #nb functional inputs
  
  ztest <- data.frame(x1 = x[,1]/nbspat[1],  # data.frame input
                      x2 = x[,2]/nbspat[2],
                      u = rep(nrep+1, each = nrow(x)))
  
  # kernType_f <- model$covariance
  
  
  ## updating list with the new functional input 
  ftest <- PCABasisComputeNew(finputs_test, finputs_train)
  # attr(ftest, "kernType") <- kernType_f
  # groupListTest <- ftest
  distffTest <- dist2FFvec(ftest$coef, ftest$coef, ftest$GramM) # precomputed distances
  ftest$distff <- distffTest
  model$covariance@covAlls$kfunc@groupList <- ftest
  
  ## predicting test data
  time2pred <- proc.time()
  pred <- predict(model, ztest, type = 'SK', lightReturn = TRUE)
  time2pred <- proc.time() - time2pred
  
  pred$mean[pred$mean < 0] <- 0
  
  ## creating data.frame with spatial locations and predictions
  output <- data.frame(x1 = x[,1], x2 = x[,2], ypred = pred$mean)
  if (return_variance == 1)
    output$variance <- pred$sdSK
  return(output)
}
