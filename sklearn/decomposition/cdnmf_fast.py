# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Author: Mathieu Blondel, Tom Dupre la Tour
# License: BSD 3 clause

from libc.math cimport fabs


def _update_cdnmf_fast(double[:, ::1] W, double[:, :] HHt, double[:, :] XHt,
                       Py_ssize_t[::1] permutation):
    violation = 0
    n_components = W.shape[1]
    n_samples = W.shape[0]  # n_features for H update

    for s in range(n_components):
        t = permutation[s]
        for i in range(n_samples):
            # gradient = GW[t, i] where GW = np.dot(W, HHt) - XHt
            grad = -XHt[i, t]

            for r in range(n_components):
                grad += HHt[t, r] * W[i, r]

            # projected gradient
            pg = min(0., grad) if W[i, t] == 0 else grad
            violation += fabs(pg)

            # Hessian
            hess = HHt[t, t]

            if hess != 0:
                W[i, t] = max(W[i, t] - grad / hess, 0.)

    return violation
