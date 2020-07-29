#### kernels for scalar inputs (si) ####
ksCompute <- function(x1, x2 = x1, par) {
  # x1, x2 = matrices with  (x1 coordinate, x2 coordinate)
  # par = (variance, lengthscale x1, lengthscale x2)
  if (is.null(attr(x1, "kernType")))
    attr(x1, "kernType") <- attr(x2, "kernType") <- "Matern52"
  
  # Computing spatial kernel
  kernType <- attr(x1, "kernType")
  kernName <- paste("ks", kernType, sep = "")
  kernFun <- try(get(kernName))
  if (class(kernFun) == "try-error") {
    stop('spatial kernel "', kernType, '" is not supported')
  } else {
    kern <- kernFun(x1, x2, par)
  }
  attr(kern, "gradient") <- attr(kern, "gradient")
  return(kern)
}

##### 2D SE kernel ####
ksGaussian <- function(x1, x2 = x1, par) {
  # x1, x2 = matrices with  (x1 coordinate, x2 coordinate)
  # par = (variance, lengthscale x1, lengthscale x2)
  sigma2 <- par[1]; theta_x1 <- par[2]; theta_x2 <- par[3]
  
  # precomputing some terms
  dist2_x1 <- outer(x1[, 1]/theta_x1, x2[, 1]/theta_x1, '-')^2
  dist2_x2 <- outer(x1[, 2]/theta_x2, x2[, 2]/theta_x2, '-')^2
  dist2 <- dist2_x1 + dist2_x2
  
  # computing the kernel
  kern <- sigma2*exp(-0.5*dist2)
  
  # computing gradients
  dkern <- Grad <- array(NA, dim = c(nrow(x1), nrow(x2), length(par)),
                         dimnames = list(rep("", nrow(x1)), rep("", nrow(x2)), names(par)))
  dkern[ , , 1] <- kern/sigma2
  dkern[ , , 2] <- kern*dist2_x1/theta_x1
  dkern[ , , 3] <- kern*dist2_x2/theta_x2
  attr(kern, "gradient") <- dkern
  # attr(kern, "gradient") <- list(sigma2 = kern/sigma2,
  #                                theta_x1 = kern*dist2_x1/theta_x1,
  #                                theta_x2 = kern*dist2_x2/theta_x2)
  return(kern)
}

##### 2D Matern 5/2 kernel ####
ksMatern52 <- function(x1, x2 = x1, par) {
  # x1, x2 = matrices with  (x1 coordinate, x2 coordinate)
  # par = (variance, lengthscale x1, lengthscale x2)
  sigma2 <- par[1]; theta_x1 <- par[2]; theta_x2 <- par[3]
  
  # precomputing some terms
  dist2_x1 <- outer(x1[, 1]/theta_x1, x2[, 1]/theta_x1, '-')^2
  dist2_x2 <- outer(x1[, 2]/theta_x2, x2[, 2]/theta_x2, '-')^2
  dist2 <- dist2_x1 + dist2_x2
  dist <- sqrt(dist2)
  sqrt5dist <- sqrt(5)*dist
  expSqrt5dist <- exp(-sqrt5dist)

  # computing the kernel
  kern <- sigma2*(1 + sqrt5dist + (5/3)*dist2)*expSqrt5dist
  
  # computing gradients
  # diag(dist) <- Inf
  dist[which(dist == 0)] = Inf
  sqrt5invDist <-  sqrt(5)/dist
  coeff <- - sigma2*(sqrt5invDist + (10/3))*expSqrt5dist + sqrt5invDist*kern
  
  dkern <- Grad <- array(NA, dim = c(nrow(x1), nrow(x2), length(par)),
                         dimnames = list(rep("", nrow(x1)), rep("", nrow(x2)), names(par)))
  dkern[ , , 1] <- kern/sigma2
  dkern[ , , 2] <- coeff*dist2_x1/theta_x1
  dkern[ , , 3] <- coeff*dist2_x2/theta_x2
  attr(kern, "gradient") <- dkern
  # attr(kern, "gradient") <- list(sigma2 = kern/sigma2,
  #                                theta_x1 = coeff*dist2_x1/theta_x1,
  #                                theta_x2 = coeff*dist2_x2/theta_x2)
  return(kern)
}

##### 2D Matern 3/2 kernel ####
ksMatern32 <- function(x1, x2 = x1, par) {
  # x1, x2 = matrices with  (x1 coordinate, x2 coordinate)
  # par = (variance, lengthscale x1, lengthscale x2)
  sigma2 <- par[1]; theta_x1 <- par[2]; theta_x2 <- par[3]
  
  # precomputing some terms
  dist2_x1 <- outer(x1[, 1]/theta_x1, x2[, 1]/theta_x1, '-')^2
  dist2_x2 <- outer(x1[, 2]/theta_x2, x2[, 2]/theta_x2, '-')^2
  dist <- sqrt(dist2_x1 + dist2_x2)
  sqrt3dist <- sqrt(3)*dist
  expSqrt3dist <- exp(-sqrt3dist)
  
  # computing the kernel
  kern <- sigma2*(1 + sqrt3dist)*expSqrt3dist
  
  # computing gradients
  diag(dist) <- Inf
  sqrt3invDist <-  sqrt(3)/dist
  coeff <- (- sigma2*expSqrt3dist + kern)*sqrt3invDist
  
  dkern <- Grad <- array(NA, dim = c(nrow(x1), nrow(x2), length(par)),
                         dimnames = list(rep("", nrow(x1)), rep("", nrow(x2)), names(par)))
  dkern[ , , 1] <- kern/sigma2
  dkern[ , , 2] <- coeff*dist2_x1/theta_x1
  dkern[ , , 3] <- coeff*dist2_x2/theta_x2
  attr(kern, "gradient") <- dkern
  # attr(kern, "gradient") <- list(sigma2 = kern/sigma2,
  #                                theta_x1 = coeff*dist2_x1/theta_x1,
  #                                theta_x2 = coeff*dist2_x2/theta_x2)
  return(kern)
}

##### 2D Exponential kernel ####
ksExponential <- function(x1, x2 = x1, par) {
  # x1, x2 = matrices with  (x1 coordinate, x2 coordinate)
  # par = (variance, lengthscale x1, lengthscale x2)
  sigma2 <- par[1]; theta_x1 <- par[2]; theta_x2 <- par[3]
  
  # precomputing some terms
  dist2_x1 <- outer(x1[, 1]/theta_x1, x2[, 1]/theta_x1, '-')^2
  dist2_x2 <- outer(x1[, 2]/theta_x2, x2[, 2]/theta_x2, '-')^2
  dist <- sqrt(dist2_x1 + dist2_x2)

  # computing the kernel
  kern <- sigma2*exp(-dist)
  
  # computing gradients
  diag(dist) <- Inf
  coeff <- kern/dist
  
  dkern <- Grad <- array(NA, dim = c(nrow(x1), nrow(x2), length(par)),
                         dimnames = list(rep("", nrow(x1)), rep("", nrow(x2)), names(par)))
  dkern[ , , 1] <- kern/sigma2
  dkern[ , , 2] <- coeff*dist2_x1/theta_x1
  dkern[ , , 3] <- coeff*dist2_x2/theta_x2
  attr(kern, "gradient") <- dkern
  # attr(kern, "gradient") <- list(sigma2 = kern/sigma2,
  #                                theta_x1 = coeff*dist2_x1/theta_x1,
  #                                theta_x2 = coeff*dist2_x2/theta_x2)
  return(kern)
}
