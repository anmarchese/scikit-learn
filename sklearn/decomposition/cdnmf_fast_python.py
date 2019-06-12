# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Author: Mathieu Blondel, Tom Dupre la Tour
# License: BSD 3 clause
from joblib import Parallel, delayed
from functools import partial
from multiprocessing import Pool

def compute_sample_stuff(i,W, HHt, XHt, permutation,t):
    # gradient = GW[t, i] where GW = np.dot(W, HHt) - XHt
    violation=0
    grad = -XHt[i, t]

    for r in range(W.shape[1]):
        grad += HHt[t, r] * W[i, r]

    # projected gradient
    pg = min(0., grad) if W[i, t] == 0 else grad
    violation += abs(pg)

    # Hessian
    hess = HHt[t, t]

    if hess != 0:
        W[i, t] = max(W[i, t] - grad / hess, 0.)

    return violation, W[i,:]


def _update_cdnmf_fast(W, HHt, XHt, permutation):
    violation = 0
    n_components = W.shape[1]
    n_samples = W.shape[0]  # n_features for H update
    for s in range(n_components):
        t = permutation[s]
        pool = Pool(4)
        par_help = partial(compute_sample_stuff, W=W,HHt=HHt,XHt=XHt,permutation=permutation,t=t)
        violations, W_temps = zip(*pool.map(par_help, range(n_samples)))
        violation += sum(violations)
        for i in range(n_samples):
            W[i,:] = W_temps[i]
        pool.close()
        pool.join()

    return violation
