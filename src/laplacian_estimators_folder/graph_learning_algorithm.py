import numpy as np
import math
# from tqdm.notebook import tqdm
from tqdm import tqdm
from numba import jit
from functools import lru_cache


def Adj(w):
    p = int((1 + math.sqrt(8 * w.size + 1)) / 2)
    idx_1, idx_2 = np.triu_indices(p, k=1)
    A = np.zeros((p, p))
    A[idx_1, idx_2] = w
    A[idx_2, idx_1] = w
    return A


def Deg(w):
    degree = Adj(w).sum(axis=0)
    return degree


def Lap(w):
    A = Adj(w)
    L = np.diag(A.sum(axis=0)) - A
    return L


def AdjStar(M):
    p = M.shape[0]
    j, l = np.triu_indices(p, k=1)
    return M[l, j] + M[j, l]
    # return w


def LapStar(M):
    p = M.shape[0]
    j, l = np.triu_indices(p, k=1)
    return M[j, j] + M[l, l] - (M[l, j] + M[j, l])


def DegStar(w):
    return LapStar(np.diag(w))


def AdjInv(A):
    idx_1, idx_2 = np.triu_indices(A.shape[0], k=1)
    w = A[idx_1, idx_2]
    return w


def LapInv(A):
    idx_1, idx_2 = np.triu_indices(A.shape[0], k=1)
    w = -A[idx_1, idx_2]
    return w


def initialize_weights(w0, p):
    if w0 is not None:
        return w0

    w = (p * (p - 1) // 2) * np.ones(p * (p - 1) // 2)
    A0 = Adj(w)
    A0 = A0 / A0.sum(axis=0, keepdims=True)
    w = AdjInv(A0)
    return w


def compute_student_weights(w, LstarSq, p, nu):
    return (p + nu) / (np.sum(w * LstarSq) + nu)


def learn_connected_graph(S, w0=None, d=1,
                          rho=1, maxiter=300,
                          reltol=1e-5, verbose=True,
                          mu=2, tau=2):
    # number of nodes
    p = S.shape[0]
    # w-initialization
    w = initialize_weights(w0, p)

    LstarS = LapStar(S)
    J = np.ones((p, p)) / p

    # Theta-initilization
    Lw = Lap(w)
    Theta = Lw
    Y = np.zeros((p, p))
    y = np.zeros(p)

    it = range(maxiter)
    if verbose:
        it = tqdm(it)

    has_converged = False

    for i in it:
        # update w
        LstarLw = LapStar(Lw)
        DstarDw = DegStar(np.diag(Lw))
        grad = LstarS - LapStar(Y + rho * Theta) + DegStar(y - rho * d) + rho * (LstarLw + DstarDw)
        eta = 1 / (2 * rho * (2 * p - 1))
        wi = w - eta * grad
        wi[wi < 0] = 0
        Lwi = Lap(wi)
        # update Theta
        gamma, V = np.linalg.eigh(rho * (Lwi + J) - Y)
        Thetai = V @ np.diag((gamma + np.sqrt(gamma ** 2 + 4 * rho)) / (2 * rho)) @ V.T - J
        # update Y
        R1 = Thetai - Lwi
        Y = Y + rho * R1
        # update y
        R2 = np.diag(Lwi) - d
        y = y + rho * R2
        # update rho
        s = rho * np.linalg.norm(LapStar(Theta - Thetai), 2)
        r = np.linalg.norm(R1, "fro")
        if r > mu * s:
            rho = rho * tau
        elif s > mu * r:
            rho = rho / tau
        error = np.linalg.norm(Lwi - Lw, 'fro') / np.linalg.norm(Lw, 'fro')

        # print(f'iter {i + 1}) Error: {error}, Sparcity: {(np.abs(wi) < 1e-5).mean()}, rho: {rho}')

        w = wi
        Lw = Lwi
        Theta = Thetai

        has_converged = (error < reltol) and (i > 1)
        if has_converged:
            break

    results = {'L': Lap(w),
               'A': Adj(w),
               'w': w,
               'maxiter': i,
               'rho': rho,
               'convergence': has_converged,
               }
    return results


# https://github.com/mirca/fingraph/blob/main/R/connected-graph-heavy-tail-admm.R

'''
learn_regular_heavytail_graph <- function(X,
                                          heavy_type = "gaussian",
                                          nu = NULL,
                                          w0 = "naive",
                                          d = 1,
                                          rho = 1,
                                          update_rho = TRUE,
                                          maxiter = 10000,
                                          reltol = 1e-5,
                                          verbose = TRUE) {
  X <- as.matrix(X)
  # number of nodes
  p <- ncol(X)
  # number of observations
  n <- nrow(X)
  LstarSq <- vector(mode = "list", length = n)
  for (i in 1:n)
    LstarSq[[i]] <- Lstar(X[i, ] %*% t(X[i, ])) / (n-1)
  # w-initialization
  w <- spectralGraphTopology:::w_init(w0, MASS::ginv(stats::cor(X)))
  A0 <- A(w)
  A0 <- A0 / rowSums(A0)
  w <- spectralGraphTopology:::Ainv(A0)
  J <- matrix(1, p, p) / p
  # Theta-initilization
  Lw <- L(w)
  Theta <- Lw
  Y <- matrix(0, p, p)
  y <- rep(0, p)
  # ADMM constants
  mu <- 2
  tau <- 2
  # residual vectors
  primal_lap_residual <- c()
  primal_deg_residual <- c()
  dual_residual <- c()
  # augmented lagrangian vector
  lagrangian <- c()
  if (verbose)
    pb <- progress::progress_bar$new(format = "<:bar> :current/:total  eta: :eta",
                                     total = maxiter, clear = FALSE, width = 80)
  elapsed_time <- c()
  start_time <- proc.time()[3]
  for (i in 1:maxiter) {
    # update w
    LstarLw <- Lstar(Lw)
    DstarDw <- Dstar(diag(Lw))
    LstarSweighted <- rep(0, .5*p*(p-1))
    if (heavy_type == "student") {
      for (q in 1:n)
        LstarSweighted <- LstarSweighted + LstarSq[[q]] * compute_student_weights(w, LstarSq[[q]], p, nu)
    } else if(heavy_type == "gaussian") {
      for (q in 1:n)
        LstarSweighted <- LstarSweighted + LstarSq[[q]]
    }
    grad <- LstarSweighted - Lstar(rho * Theta + Y) + Dstar(y - rho * d) + rho * (LstarLw + DstarDw)
    eta <- 1 / (2*rho * (2*p - 1))
    wi <- w - eta * grad
    wi[wi < 0] <- 0
    Lwi <- L(wi)
    # update Theta
    eig <- eigen(rho * (Lwi + J) - Y, symmetric = TRUE)
    V <- eig$vectors
    gamma <- eig$values
    Thetai <- V %*% diag((gamma + sqrt(gamma^2 + 4 * rho)) / (2 * rho)) %*% t(V) - J
    # update Y
    R1 <- Thetai - Lwi
    Y <- Y + rho * R1
    # update y
    R2 <- diag(Lwi) - d
    y <- y + rho * R2
    # compute primal, dual residuals, & lagrangian
    primal_lap_residual <- c(primal_lap_residual, norm(R1, "F"))
    primal_deg_residual <- c(primal_deg_residual, norm(R2, "2"))
    dual_residual <- c(dual_residual, rho*norm(Lstar(Theta - Thetai), "2"))
    lagrangian <- c(lagrangian, compute_augmented_lagrangian_ht(wi, LstarSq, Thetai, J, Y, y, d, heavy_type, n, p, rho, nu))
    # update rho
    if (update_rho) {
      s <- rho * norm(Lstar(Theta - Thetai), "2")
      r <- norm(R1, "F")
      if (r > mu * s)
        rho <- rho * tau
      else if (s > mu * r)
        rho <- rho / tau
    }
    if (verbose)
      pb$tick()
    has_converged <- (norm(Lw - Lwi, 'F') / norm(Lw, 'F') < reltol) && (i > 1)
    elapsed_time <- c(elapsed_time, proc.time()[3] - start_time)
    if (has_converged)
      break
    w <- wi
    Lw <- Lwi
    Theta <- Thetai
  }
  results <- list(laplacian = L(wi),
                  adjacency = A(wi),
                  theta = Thetai,
                  maxiter = i,
                  convergence = has_converged,
                  primal_lap_residual = primal_lap_residual,
                  primal_deg_residual = primal_deg_residual,
                  dual_residual = dual_residual,
                  lagrangian = lagrangian,
                  elapsed_time = elapsed_time)
  return(results)
}

compute_student_weights <- function(w, LstarSq, p, nu) {
  return((p + nu) / (sum(w * LstarSq) + nu))
}

compute_augmented_lagrangian_ht <- function(w, LstarSq, Theta, J, Y, y, d, heavy_type, n, p, rho, nu) {
  eig <- eigen(Theta + J, symmetric = TRUE, only.values = TRUE)$values
  Lw <- L(w)
  Dw <- diag(Lw)
  u_func <- 0
  if (heavy_type == "student") {
    for (q in 1:n)
      u_func <- u_func + (p + nu) * log(1 + n * sum(w * LstarSq[[q]]) / nu)
  } else if (heavy_type == "gaussian"){
    for (q in 1:n)
      u_func <- u_func + sum(n * w * LstarSq[[q]])
  }
  u_func <- u_func / n
  return(u_func - sum(log(eig)) + sum(y * (Dw - d)) + sum(diag(Y %*% (Theta - Lw)))
         + .5 * rho * (norm(Dw - d, "2")^2 + norm(Lw - Theta, "F")^2))
}
'''


# @lru_cache(maxsize=None)
def learn_connected_graph_heavy_tails(X_mat, heavy_type="student",
                                      is_covariance=False,
                                      normalize=True,
                                      nu=3, w0=None,
                                      d=1, rho=1,
                                      update_rho=True,
                                      max_iter=300,
                                      tol=1e-5,
                                      verbose=True,
                                      mu=2, tau=2):
    # number of nodes
    p = X_mat.shape[1]
    # number of observations
    n = X_mat.shape[0]

    # normalize X_mat
    if normalize:
        X_mat = (X_mat - X_mat.mean(axis=0)) / X_mat.std(axis=0)

    # w-initialization
    w = initialize_weights(w0, p)

    # L-star initialization
    if is_covariance:
        LstarSweighted = LapStar(X_mat)
    else:
        LstarSq = [LapStar(X_mat[[i]].T @ X_mat[[i]]) / (n - 1)
                   for i in range(n)]

        if heavy_type == "student":
            LstarSweighted = sum(map(lambda LstarSq_i: LstarSq_i * compute_student_weights(w, LstarSq_i, p, nu),
                                     LstarSq))
        # elif heavy_type == "gaussian":
        else:
            LstarSweighted = sum(LstarSq)

    # print('L Star Correlation Matrix:\n', LapStar(np.corrcoef(X_mat.T)), '\n')
    # print('L Star Covariance Matrix:\n', LapStar(X_mat.T @ X_mat), '\n')
    # print('L Star Weighted:\n', LstarSweighted, '\n')
    # print('L Star Gaussian Weighted:\n', sum(LstarSq), '\n')
    # print('L Star Correlation Matrix mean:\n', LapStar(np.corrcoef(X_mat.T)).mean(), '\n')
    # print('L Star Weighted mean:', LstarSweighted.mean(), '\n')
    # print('L Star Weighted Rescaled:\n', LstarSweighted, '\n')

    J = np.ones((p, p)) / p

    # Theta-initilization
    Lw = Lap(w)
    Theta = Lw
    Y = np.zeros((p, p))
    y = np.zeros(p)

    it = range(max_iter)
    if verbose:
        it = tqdm(it)

    has_converged = False

    for i in it:
        # update w
        LstarLw = LapStar(Lw)
        DstarDw = DegStar(np.diag(Lw))

        if not is_covariance and heavy_type == "student" and i % 50 == 0:
            LstarSweighted = sum(map(lambda LstarSq_i: LstarSq_i * compute_student_weights(w, LstarSq_i, p, nu),
                                     LstarSq))
            # print(f'{i}) L Star Weighted: {LstarSweighted}')

        grad = LstarSweighted - LapStar(Y + rho * Theta) + DegStar(y - rho * d) + rho * (LstarLw + DstarDw)
        eta = 1 / (2 * rho * (2 * p - 1))
        wi = w - eta * grad
        wi[wi < 0] = 0
        Lwi = Lap(wi)

        # update Theta
        gamma, V = np.linalg.eigh(rho * (Lwi + J) - Y)
        Thetai = V @ np.diag((gamma + np.sqrt(gamma ** 2 + 4 * rho)) / (2 * rho)) @ V.T - J

        # update Y
        R1 = Thetai - Lwi
        Y = Y + rho * R1

        # update y
        R2 = np.diag(Lwi) - d
        y = y + rho * R2

        # update rho
        if update_rho:
            s = rho * np.linalg.norm(LapStar(Theta - Thetai), 2)
            r = np.linalg.norm(R1, "fro")
            if r > mu * s:
                rho = rho * tau
            elif s > mu * r:
                rho = rho / tau

        error = np.linalg.norm(Lwi - Lw, 'fro') / np.linalg.norm(Lw, 'fro')

        w = wi
        Lw = Lwi
        Theta = Thetai

        has_converged = (error < tol) and (i > 1)
        if has_converged:
            break

    results = {'L': Lap(w),
               'A': Adj(w),
               'w': w,
               'maxiter': i,
               'rho': rho,
               'convergence': has_converged,
               }
    return results
