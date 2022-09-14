#### Computation of the Basis functions ####
basisCompute <- function(f, method = c("PCA"), nbterms = NULL, ...) {
  # method = projection method onto basis functions
  # nbterms = number of terms used in the projection
  # ... = additional terms to be passed
  
  method <- match.arg(method)
  
  # # Computing basis
  # projName <- paste(method, "BasisCompute", sep = "")
  # projFun <- try(get(projName))
  # if (class(projFun) == "try-error") {
  #   stop('projection method "', method, '" is not supported')
  # } else {
  #   basis <- projFun(f, nbterms, ...)
  #   attr(basis, "projType") <- method
  # }
  # return(basis)
  
  # Computing basis
  projFun <- get(paste(method, "BasisCompute", sep = ""))
  basis <- projFun(f, nbterms, ...)
  attr(basis, "projType") <- method
  return(basis)
}

PCABasisCompute <- function(f, nbterms = NULL, lv.inertia = 0.97) {
  # nbterms = number of terms used in the projection
  # f = list with replicates of the functional input (nb replicates x length time serie)
  # ... not required
  nf <- length(f)
  nrep <- nrow(f[[1]])
  
  # # Computing PCA basis and coefficients
  # eigVects <- lapply(lapply(f, cov), function(x) {eigen(x)$vectors})
  # truncEigVects <- lapply(eigVects, "[", T, 1:nbterms[1])
  # names(truncEigVects) <- paste("f", 1:nf, sep = "")
  # coeff <- array(NaN, dim = c(nbterms[1], nf, nrep))
  # for (i in 1:length(f))
  #   coeff[ , i, ] <- t(truncEigVects[[i]]) %*% t(f[[i]])
  # coeff <- lapply(seq(dim(coeff)[3]), function(x) coeff[ , , x])
  # names(coeff) <- paste("F", 1:nrep, sep = "")
  
  # Computing PCA basis and coefficients
  if (is.null(nbterms)) {
    eigDecomp <- lapply(lapply(f, cov), function(x) {eigen(x)})
    eigVals <- lapply(eigDecomp, "[[", 1)
    eigVects <- lapply(eigDecomp, "[[", 2)

    # nbterms <- lapply(eigVals, function(x) {which(cumsum(x^2)/sum(x^2) >= lv.inertia)[1]})
    nbterms <- lapply(eigVals, function(x) {which(cumsum(x)/sum(x) >= lv.inertia)[1]})
  } else {
    eigVects <- lapply(lapply(f, cov), function(x) {eigen(x)$vectors})
    if (length(nbterms) == 1) {
      nbterms <- rep(nbterms, nf)
    } else if (length(nbterms) != nf) {
      stop("The length of nbterms should be equal to the number of functional inputs")
    }
  }

  truncEigVects <- Map("[", lapply(eigVects, "t"), lapply(nbterms, seq), simplified = T)
  truncEigVects <- lapply(seq(nf),
                          function(x) {
                            if (nbterms[[x]] == 1)
                              truncEigVects[[x]] <- matrix(truncEigVects[[x]], nrow = 1);
                            return(t(truncEigVects[[x]]))
                            })
  names(truncEigVects) <- paste("f", 1:nf, sep = "")

  coeff <- Map("%*%", lapply(truncEigVects, "t"), lapply(f, "t"))
  attributes(coeff) <- NULL
  # coeff <- lapply(seq(nrep) , function(x) {simplify2array(lapply(coeff, "[", T, x))})
  coeff <- lapply(seq(nrep) , function(x) {lapply(coeff, "[", T, x)})
  names(coeff) <- paste("F", 1:nrep, sep = "")
  
  # passing list with key elemements of the projection (coeff, basis, Gram matrix)
  projBasis <- vector("list", 4)
  names(projBasis) = c("f", "coef", "basis", "GramM")
  projBasis$f <- f
  projBasis$coef <- coeff
  projBasis$basis <- truncEigVects
  # projBasis$GramM <- lapply(seq(nf), function(x) diag(nbterms))
  projBasis$GramM <- lapply(seq(nf), function(x) diag(nbterms[x]))
  names(projBasis$GramM) <- names(truncEigVects)
  return(projBasis)
}

#### Computation of the Basis functions ####
basisComputeNew <- function(fnew, f, ...) {
  # fnew = list with new functional inputs
  # f = list with old functional inputs
  # ... = additional terms to be passed
  method <- attr(f, "projType")

  # # Computing basis
  # projName <- paste(method, "BasisCompute", sep = "")
  # projFun <- try(get(projName))
  # if (class(projFun) == "try-error") {
  #   stop('projection method "', method, '" is not supported')
  # } else {
  #   basis <- projFun(f, nbterms, ...)
  #   attr(basis, "projType") <- method
  # }
  # return(basis)
  
  # Computing basis
  projFun <- get(paste(method, "BasisComputeNew", sep = ""))
  basis <- projFun(fnew, f, ...)
  attr(basis, "projType") <- method
  return(basis)
}


PCABasisComputeNew <- function(fnew, f) {
  # nbterms = number of terms used in the projection
  # f = list with replicates of the functional input (nb replicates x length time serie)
  # ... not required
  nf <- length(f$f)
  nt <- ncol(f$f[[1]])
  
  fnew <- lapply(fnew, function(x) as.matrix(x))
  if(ncol(fnew[[1]]) == 1)
    fnew <- lapply(fnew, "t")
  
  nrep <- nrow(f$f[[1]])
  nrepNew <- nrow(fnew[[1]])
  
  # passing parameters of the projection to be used
  coeffNew <- Map("%*%", lapply(f$basis, "t"), lapply(fnew, "t"))
  attributes(coeffNew) <- NULL
  coeffNew <- lapply(seq(nrepNew) , function(x) {lapply(coeffNew, "[", T, x)})
  names(coeffNew) <- paste("F", nrep + 1:nrepNew, sep = "")
  
  # passing list with key elemements of the projection (coeff, basis, Gram matrix)
  projBasis <- vector("list", 4)
  names(projBasis) = c("f", "coef", "basis", "GramM")
  projBasis$f <- lapply(1:nf, function(x) rbind(f$f[[x]], fnew[[x]]))
  projBasis$coef <- c(f$coef, coeffNew)
  projBasis$basis <- f$basis
  projBasis$GramM <- f$GramM
  return(projBasis)
}



splinesBasisCompute <- function(nbterms, f) {
  # nbterms = number of terms used in the projection
  # f = replicates of the functional input (nb replicates x length time serie)
  
  require(splines)
  # require(matlib)
  
  # Computing splines basis
  projBasis <- vector("list", 3)
  names(projBasis) = c("coef", "basis", "GramM")
  
  #☻ in progress...
  
  return(projBasis)
}


# # **************************************************************
# genBasSplines <- function(k, nt, ord){
#   ll = 1
#   ul = 37
#   addknot <- ord-1
#   knots <- rep(ll,addknot)
#   knots <- c(knots,rep(ul,addknot))
#   basknot <- k-ord+2
#   knots <- sort(c(knots,seq(ll, ul, length.out = basknot)))
#   testers <- seq(ll, ul, length.out = nt)
#   basis <- splineDesign(knots = knots, x = ll:ul, outer.ok = T, ord = ord)
# }
# # **************************************************************
