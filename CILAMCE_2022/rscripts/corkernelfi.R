#### kernels for functional inputs (si) ####
kfCompute <- function(F1, F2 = F1, par) {
  # F1, F2 = list of realisations with functional inputs (coefficients, basis, Psi, ...)
  # par = (lengthscales)
  if (is.null(attr(F1, "kernType")))
    attr(F1, "kernType") <- attr(F2, "kernType") <- "Matern52"

  # Computing the kernel for the functional inputs
  kernType <- attr(F1, "kernType")
  kernName <- paste("kf", kernType, sep = "")
  kernFun <- try(get(kernName))
  if (class(kernFun) == "try-error") {
    stop('kernel "', kernType, '" is not supported for functional inputs')
  } else {
    kern <- kernFun(F1, F2, par)
    rownames(kern) <- names(F1$coef)
    colnames(kern) <- names(F2$coef)
  }
  attr(kern, "gradient") <- attr(kern, "gradient")
  return(kern)
}
 
# dist2ffvec <- function(f1, f2 = f1, Psi) {
#   # f1, f2 = list with functional inputs (coefficients)
#   # Psi = integrated Gram matrix of the basis
#   # nf <- ncol(f1)
#   nf <- length(f1)
#   dcoeff <- Map("-", f1, f2)
#   dist2ff <- sapply(seq(nf),
#                     function(x) {diag(t(dcoeff[[x]]) %*% Psi[[x]] %*% dcoeff[[x]])})
#   return(dist2ff)
# }
# 
# dist2FFvec <- function(F1, F2 = F1, Psi) {
#   # F1, F2 = list with realisations of functional inputs (coefficients, basis)
#   # Psi = Gram matrix of the basis
#   nfs1 <- length(F1)
#   nfs2 <- length(F2)
#   dist2 <- vector("list", nfs1)
#   for (i in 1:nfs1) {
#     temp <- vector("list", nfs2)
#     for (j in 1:nfs2) {
#       temp[[j]] <- dist2ffvec(F1[[i]], F2[[j]], Psi)
#     }
#     dist2[[i]] <- temp
#   }
#   return(dist2)
# }
# 
# dist2f_scaled <- function(f1, f2 = f1, theta, Psi, preDist2ff = NULL) {
#   # f1, f2 = list with functional inputs (coefficients, basis)
#   # Psi = integrated Gram matrix of the basis
#   # theta = lengthscale parameters
#   # preDist2ff = precomputed distances
# 
#   # nf <- ncol(f1)
#   nf <- length(f1)
#   if (length(theta) != nf)
#     stop('inconsistent dimensions')
# 
#   if (!is.null(preDist2ff)) {
#     dist2ff <- preDist2ff/theta^2
#   } else {
#     # dcoeff <- f1 - f2
#     dcoeff <- Map("-", f1, f2)
#     # # dist2ff <- diag(t(dcoeff) %*% Psi %*% dcoeff)/theta^2
#     # dist2ff <- rep(NaN, nf)
#     # for (i in 1:nf)
#     #   dist2ff[i] <- diag(t(dcoeff[,i]) %*% Psi[[i]] %*% dcoeff[,i])/theta[i]^2
#     dist2ff <- sapply(seq(nf),
#                       function(x) {diag(t(dcoeff[[x]]) %*% Psi[[x]] %*% dcoeff[[x]])/theta[x]^2})
#   }
# 
#   # computing the total distance
#   dist2 <- sum(dist2ff)
# 
#   # computing gradients
#   ddist2ff <- -2*dist2ff/theta
#   names(ddist2ff) <- names(theta)
#   attr(dist2, "gradient") <- ddist2ff
#   # dimf <- length(theta)
#   # hfun <- parse(text = paste("list(", paste("theta_f", 1:dimf, " = ddist2ff[",
#   #                              1:dimf, "]", sep = "", collapse = ", "), ")"))
#   # attr(dist2, "gradient") <- eval(hfun)
#   return(dist2)
# }
# 
# dist2F_scaled <- function(F1, F2 = F1, theta, Psi, distFF = NULL) {
#   # F1, F2 = list with realisations of functional inputs (coefficients, basis)
#   # Psi = Gram matrix of the basis
#   # theta = lengthscale parameters
#   # distFF = list with precomputed distances
#   nfs1 <- length(F1)
#   nfs2 <- length(F2)
#   dimf <- length(theta)
#   dist2 <- matrix(0, nfs1, nfs2)
#   ddist2 <- array(NA, dim = c(nfs1, nfs2, dimf),
#                   dimnames = list(names(F1), names(F2), names(theta)))
#   # ddist2 <- array(0, dim = c(nfs1, nfs2, dimf))
#   for (i in 1:nfs1) {
#     for (j in 1:nfs2) {
#       if (!is.null(distFF)) {
#         temp <- dist2f_scaled(F1[[i]], F2[[j]], theta, Psi, distFF[[i]][[j]])
#       } else {
#         temp <- dist2f_scaled(F1[[i]], F2[[j]], theta, Psi)
#       }
# 
#       dist2[i, j] <- temp
#       # for (k in 1:dimf)
#       #   ddist2[i, j, k] <- attr(temp, "gradient")[k]
#       ddist2[i, j, ] <- attr(temp, "gradient")
#       # ddist2[i, j, k] <- attr(temp, "gradient")[[k]]
#     }
#   }
#   attr(dist2, "gradient") <- ddist2
#   # hfun <- parse(text = paste("list(", paste("theta_f", 1:dimf, " = ddist2[ , , ",
#   #                                          1:dimf, "]", sep = "", collapse = ", "), ")"))
#   # attr(dist2, "gradient") <- eval(hfun)
#   return(dist2)
# }

##### SE kernel ####
kfGaussian <- function(F1, F2 = F1, par) {
  # F1, F2 = list of realisations with functional inputs (coefficients, basis, Psi)
  # par = (lengthscales)
  theta <- par
  # Psi <- attr(F1, "Psi") # Psi = integrated Gram matrix of the basis
  Psi <- F1$GramM
  if ('distff' %in% names(F1) & !is.null(F1$distff)) {
    distff <- F1$distff
    preCompDist <- TRUE
  } else {
    preCompDist <- FALSE
  }
  F1 <- F1$coef; F2 <- F2$coef

  # precomputing some terms
  if (preCompDist) {
    dist2 <- dist2F_scaled(F1, F2, theta, Psi, distff)
  } else {
    dist2 <- dist2F_scaled(F1, F2, theta, Psi)
  }

  # computing the kernel
  kern <- exp(-0.5*dist2)
  rownames(kern) <- names(F1)
  colnames(kern) <- names(F2)
  attr(kern, "gradient") <- NULL

  # computing gradients
  coeff <- -0.5*kern
  ddist2 <- attr(dist2, "gradient")

  # dkern <- array(NA, dim = c(length(F1), length(F2), length(par)),
  #                dimnames = list(rep("", length(F1)), rep("",length(F2)), names(par)))
  # dkern[ , , 1] <- kern/sigma2
  # dkern[ , , -1] <- array(coeff, dim(ddist2))*ddist2
  dkern <- array(coeff, dim(ddist2))*ddist2
  dimnames(dkern) = list(names(F1), names(F2), names(theta))
  attr(kern, "gradient") <- dkern

  # dimf <- length(theta)
  # hfun <- parse(text = paste("list(sigma2 = kern/sigma2, ",
  #                            paste("theta_f", 1:dimf, " = coeff*ddist2[[ ", 1:dimf, "]]",
  #                                  sep = "", collapse = ", "), ")"))
  # attr(kern, "gradient") <- eval(hfun)
  return(kern)
}

##### Matern 5/2 kernel ####
kfMatern52 <- function(F1, F2 = F1, par) {
  # F1, F2 = list of realisations with functional inputs (coefficients, basis, Psi)
  # par = (lengthscales)
  theta <- par
  # Psi <- attr(F1, "Psi") # Psi = integrated Gram matrix of the basis
  Psi <- F1$GramM
  if ('distff' %in% names(F1) & !is.null(F1$distff)) {
    distff <- F1$distff
    preCompDist <- TRUE
  } else {
    preCompDist <- FALSE
  }
  F1 <- F1$coef; F2 <- F2$coef

  # precomputing some terms
  if (preCompDist) {
    dist2 <- dist2F_scaled(F1, F2, theta, Psi, distff)
  } else {
    dist2 <- dist2F_scaled(F1, F2, theta, Psi)
  }
  dist <- sqrt(dist2)
  attr(dist, "gradient") <- NULL
  sqrt5dist <- sqrt(5)*dist
  expSqrt5dist <- exp(-sqrt5dist)

  # computing the kernel
  kern <- (1 + sqrt5dist + (5/3)*dist2)*expSqrt5dist
  rownames(kern) <- names(F1)
  colnames(kern) <- names(F2)
  attr(kern, "gradient") <- NULL

  # computing gradients
  dist[which(dist == 0)] = Inf
  # diag(dist) <- Inf
  sqrt5invDist <-  sqrt(5)/dist
  coeff <- -0.5*(- (sqrt5invDist + (10/3))*expSqrt5dist + sqrt5invDist*kern)
  ddist2 <- attr(dist2, "gradient")

  # dkern <- array(NA, dim = c(length(F1), length(F2), length(par)),
  #                dimnames = list(rep("", length(F1)), rep("",length(F2)), names(par)))
  # dkern[ , , 1] <- kern/sigma2
  # dkern[ , , -1] <- array(coeff, dim(ddist2))*ddist2
  dkern <- array(coeff, dim(ddist2))*ddist2
  dimnames(dkern) = list(names(F1), names(F2), names(theta))
  attr(kern, "gradient") <- dkern

  # dimf <- length(theta)
  # hfun <- parse(text = paste("list(sigma2 = kern/sigma2, ",
  #                            paste("theta_f", 1:dimf, " = coeff*ddist2[[ ", 1:dimf, "]]",
  #                                  sep = "", collapse = ", "), ")"))
  # attr(kern, "gradient") <- eval(hfun)
  return(kern)
}

##### Matern 3/2 kernel ####
kfMatern32 <- function(F1, F2 = F1, par) {
  # F1, F2 = list of realisations with functional inputs (coefficients, basis, Psi)
  # par = (lengthscales)
  theta <- par
  # Psi <- attr(F1, "Psi") # Psi = integrated Gram matrix of the basis
  Psi <- F1$GramM
  if ('distff' %in% names(F1) & !is.null(F1$distff)) {
    distff <- F1$distff
    preCompDist <- TRUE
  } else {
    preCompDist <- FALSE
  }
  F1 <- F1$coef; F2 <- F2$coef

  # precomputing some terms
  if (preCompDist) {
    dist2 <- dist2F_scaled(F1, F2, theta, Psi, distff)
  } else {
    dist2 <- dist2F_scaled(F1, F2, theta, Psi)
  }
  dist <- sqrt(dist2)
  attr(dist, "gradient") <- NULL
  sqrt3dist <- sqrt(3)*dist
  expSqrt3dist <- exp(-sqrt3dist)
  diag(dist) <- Inf
  sqrt3invDist <-  sqrt(3)/dist

  # computing the kernel
  kern <- (1 + sqrt3dist)*expSqrt3dist
  rownames(kern) <- names(F1)
  colnames(kern) <- names(F2)
  attr(kern, "gradient") <- NULL

  # computing gradients
  coeff <- -0.5*(- expSqrt3dist + kern)*sqrt3invDist
  ddist2 <- attr(dist2, "gradient")

  # dkern <- array(NA, dim = c(length(F1), length(F2), length(par)),
  #                dimnames = list(rep("", length(F1)), rep("",length(F2)), names(par)))
  # dkern[ , , 1] <- kern/sigma2
  # dkern[ , , -1] <- array(coeff, dim(ddist2))*ddist2
  dkern <- array(coeff, dim(ddist2))*ddist2
  dimnames(dkern) = list(names(F1), names(F2), names(theta))
  attr(kern, "gradient") <- dkern

  # dimf <- length(theta)
  # hfun <- parse(text = paste("list(sigma2 = kern/sigma2, ",
  #                            paste("theta_f", 1:dimf, " = coeff*ddist2[[ ", 1:dimf, "]]",
  #                                  sep = "", collapse = ", "), ")"))
  # attr(kern, "gradient") <- eval(hfun)
  return(kern)
}

##### Exponential kernel ####
kfExponential <- function(F1, F2 = F1, par) {
  # F1, F2 = list of realisations with functional inputs (coefficients, basis, Psi)
  # par = (lengthscales)
  theta <- par
  # Psi <- attr(F1, "Psi") # Psi = integrated Gram matrix of the basis
  Psi <- F1$GramM
  if ('distff' %in% names(F1) & !is.null(F1$distff)) {
    distff <- F1$distff
    preCompDist <- TRUE
  } else {
    preCompDist <- FALSE
  }
  F1 <- F1$coef; F2 <- F2$coef

  # precomputing some terms
  if (preCompDist) {
    dist2 <- dist2F_scaled(F1, F2, theta, Psi, distff)
  } else {
    dist2 <- dist2F_scaled(F1, F2, theta, Psi)
  }
  dist <- sqrt(dist2)
  attr(dist, "gradient") <- NULL

  # computing the kernel
  kern <- exp(-dist)
  rownames(kern) <- names(F1)
  colnames(kern) <- names(F2)
  attr(kern, "gradient") <- NULL

  # computing gradients
  diag(dist) <- Inf
  coeff <- -0.5*kern/dist
  ddist2 <- attr(dist2, "gradient")

  # dkern <- array(NA, dim = c(length(F1), length(F2), length(par)),
  #                dimnames = list(rep("", length(F1)), rep("",length(F2)), names(par)))
  # dkern[ , , 1] <- kern/sigma2
  # dkern[ , , -1] <- array(coeff, dim(ddist2))*ddist2
  dkern <- array(coeff, dim(ddist2))*ddist2
  dimnames(dkern) = list(names(F1), names(F2), names(theta))
  attr(kern, "gradient") <- dkern

  # dimf <- length(theta)
  # hfun <- parse(text = paste("list(sigma2 = kern/sigma2, ",
  #                            paste("theta_f", 1:dimf, " = coeff*ddist2[[ ", 1:dimf, "]]",
  #                                  sep = "", collapse = ", "), ")"))
  # attr(kern, "gradient") <- eval(hfun)
  return(kern)
}
