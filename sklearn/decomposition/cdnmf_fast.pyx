# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Author: Mathieu Blondel, Tom Dupre la Tour
# License: BSD 3 clause

cimport cython
from libc.math cimport fabs
from cython.parallel import prange
import numpy as np

def _update_cdnmf_fast(double[:, ::1] W, double[:, :] HHt, double[:, :] XHt,
                       Py_ssize_t[::1] permutation, int n_jobs):
    cdef double violation = 0
    cdef Py_ssize_t n_components = W.shape[1]
    cdef Py_ssize_t n_samples = W.shape[0]  # n_features for H update
    cdef double grad, pg, hess
    cdef Py_ssize_t i, r, s, t
    cdef double[:] W_i = np.zeros((n_samples,))

    if n_jobs == 1:
      with nogil:
        violation=0
        for s in range(n_components):
            t = permutation[s]

            for i in range(n_samples):
                # gradient = GW[t, i] where GW = np.dot(W, HHt) - XHt
                grad = -XHt[i, t]

                for r in range(n_components):
                    grad = grad + HHt[t, r] * W[i, r]

                # projected gradient
                pg = min(0., grad) if W[i, t] == 0 else grad
                violation += fabs(pg)

                # Hessian
                hess = HHt[t, t]

                if hess != 0:
                    W[i, t] = max(W[i, t] - grad / hess, 0.)

      return violation
    else:

      with nogil:
        violation=0
        for s in range(n_components):
            t = permutation[s]

            for i in prange(n_samples,num_threads=n_jobs):
                # gradient = GW[t, i] where GW = np.dot(W, HHt) - XHt
                grad = -XHt[i, t]

                for r in range(n_components):
                    grad = grad + HHt[t, r] * W[i, r]

                # projected gradient
                pg = min(0., grad) if W[i, t] == 0 else grad
                violation += fabs(pg)

                # Hessian
                hess = HHt[t, t]

                if hess != 0:
                    W_i[i] = max(W[i, t] - grad / hess, 0.)
                else:
                    W_i[i] = W[i, t]

            for i in range(n_samples):
                W[i,t] = W_i[i]


      return violation
