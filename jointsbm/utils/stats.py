import numpy as np
import pandas as pd
import scipy as sp
from numpy.linalg import cholesky, eig, inv, svd
from scipy.sparse import csr_matrix


def gen_design_matrix(arr, cats=None, **kwargs):
    nrow = len(arr)
    if cats is None:
        cats = np.unique(arr)
    ncol = len(cats)
    cats_dict = dict(zip(cats, range(ncol)))
    M = np.zeros([nrow, ncol], **kwargs)
    for i in range(nrow):
        M[i, cats_dict[arr[i]]] = 1.0
    return M


def factor(arr, fullRank=True, **kwargs):
    M = gen_design_matrix(arr, **kwargs)
    xx = [True] * M.shape[1]
    if fullRank:
        xx[0] = False
    return M[:, xx]


def as_dt(vec, cols=None):
    if len(vec.shape) == 1:
        vec = vec.reshape([-1, 1])
    ans = pd.DataFrame(vec)
    if cols is None:
        ans.columns = ["col{}".format(i + 1) for i in range(vec.shape[1])]
    else:
        ans.columns = cols
    return ans


def getStats(x, An):
    if sp.sparse.issparse(An):
        An = An.todense()
    An_collapsed = np.matmul(np.matmul(x.T, np.tril(An)), x)
    An_collapsed = An_collapsed + An_collapsed.T
    An_collapsed = An_collapsed - np.diag(np.diag(An_collapsed)) * 1.0 / 2
    mm = (x > 0).sum(axis=0).reshape([-1, 1])
    N = np.matmul(mm, mm.T)
    np.fill_diagonal(N, mm * (mm - 1) / 2)
    return An_collapsed, N


def estimateTheta(X, A):
    if isinstance(X, list):
        An_c, N_c = np.sum([getStats(X[i], A[i]) for i in range(len(X))], axis=0)
    else:
        An_c, N_c = getStats(X, A)
    ans = An_c * 1.0 / N_c
    # ans = np.zeros(N_c.shape)
    # ans[N_c > 0.] = An_c[N_c > 0.]*1./N_c[N_c > 0.]
    return ans


def onehot_vec(arg, K):
    ans = np.zeros([K])
    ans[arg] = 1.0
    return ans


def frobenius_norm(M):
    return np.sqrt(np.trace(np.matmul(M.T, M)))


def inv_s(M):
    diag_m = 1 / np.diag(M)
    diag_m[diag_m == float("inf")] = 0
    return np.diag(diag_m)
