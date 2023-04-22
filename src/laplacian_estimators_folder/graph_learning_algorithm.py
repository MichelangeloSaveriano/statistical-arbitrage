import numpy as np
import math
from tqdm import tqdm


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


def learn_connected_graph_heavy_tails(X_mat, heavy_type="student",
                                      is_covariance=False,
                                      normalize=True,
                                      nu=3, w0=None,
                                      d=1, rho=1,
                                      update_rho=True,
                                      max_iter=300,
                                      tol=2e-4,
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
    w0 = w

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

        if False:  # i < 10 or i % 10 == 0:
            print(f'{i}) wi: {wi}')
            print(f'wi - w0 = {wi - w0}')


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

        if False:  # i < 10 or i % 5 == 0:
            print(f'\n\nIteration {i}:')
            print(f'{i}) wi: {wi}')
            print(f'w0 = {w0}')
            print(f'LstarLw: {LstarSweighted}')
            print(f'LstarLw: {LstarLw}')
            print(f'DstarDw: {DstarDw}')
            print(f'grad: {grad}')
            # print(f'R1: {R1}')
            # print(f'R2: {R2}')
            print(f'eta: {eta}, rho: {rho}, error: {error}, |wi|: {np.linalg.norm(wi, 2)}, |wi-w0|: {np.linalg.norm(wi-w0, 2)}')

        w = wi
        Lw = Lwi
        Theta = Thetai

        has_converged = (error < tol) and (i > 1)
        if has_converged:
            break

    # print(f'N Iter: {i}')
    results = {'L': Lap(w),
               'A': Adj(w),
               'w': w,
               'maxiter': i,
               'rho': rho,
               'convergence': has_converged,
               }
    return results
