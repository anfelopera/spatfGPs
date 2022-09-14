#' Solves a triangular system with a Kronecker product structure
#'
#' Solves \eqn{kron(A, B) x = y} where \eqn{A} and \eqn{B} are lower triangular
#' matrices.
#'
#' @export
#' 
#' @param A an \eqn{m x n} matrix
#' @param B an \eqn{p x q} matrix
#' @param y an \eqn{mp x s} matrix
#' 
#' @return x
#' 
forwardsolve.kron <- function(A, B, y) {
  y <- as.matrix(y)
  n <- ncol(A)
  p <- nrow(B)
  q <- ncol(B)
  s <- ncol(y)
  
  res = matrix(NA, nrow = n*q, ncol = s)
  
  for(i in 1:s) {
    Y = matrix(y[,i], nrow = p)
    Xp = forwardsolve(B, Y)
    res[,i] = as.numeric(t(forwardsolve(A, t(Xp))))
  }
  return(res)
}

#' Solves a triangular system with a Kronecker product structure
#'
#' Solves \eqn{kron(A, B) x = y} where \eqn{A} and \eqn{B} are upper triangular
#' matrices.
#'
#' @export
#' 
#' @param A an \eqn{m x n} matrix
#' @param B an \eqn{p x q} matrix
#' @param y an \eqn{mp x s} matrix
#' 
#' @return x
#' 
backsolve.kron <- function(A, B, y) {
  y <- as.matrix(y)
  n <- ncol(A)
  p <- nrow(B)
  q <- ncol(B)
  s <- ncol(y)
  
  res <- matrix(NA, nrow = n*q, ncol = s)
  
  for(i in 1:s) {
    Y <- matrix(y[,i], nrow = p)
    Xp <- backsolve(B, Y)
    res[,i] <- as.numeric(t(backsolve(A, t(Xp))))
  }
  return(res)
}

#' Product Kronecker matrices x vector
#'
#' Solves \eqn{kron(A, B) x = y} where \eqn{A} and \eqn{B} are matrices
#' matrices.
#'
#' @export
#' 
#' @param A an \eqn{m x n} matrix
#' @param B an \eqn{p x q} matrix
#' @param x an \eqn{nq x 1} vector
#' 
#' @return y
#' 
prod.KronVec <- function(A, B, x) {
  x <- as.matrix(x)
  m <- nrow(A)
  n <- ncol(A)
  p <- nrow(B)
  q <- ncol(B)
  s <- ncol(x)

  X <- matrix(x, nrow = q, ncol = n)
  Y <- B %*% (X %*% t(A))
  
  res <- matrix(Y, nrow = p*m, ncol = s)
  return(res)
}

