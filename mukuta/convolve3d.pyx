from __future__ import division
import numpy as np
cimport numpy as np

def convolve3d(np.ndarray[np.float_t,ndim=3] ivector,np.ndarray[np.float_t,ndim=4] weight, np.int_t imax,np.int_t jmax,np.int_t kmax,np.int_t sj,np.int_t sk):
    cdef int i,j,k
    cdef np.ndarray[np.float_t,ndim=3] ovector=np.zeros((imax,jmax,kmax))
    cdef int pj=weight.shape[2]
    cdef int pk=weight.shape[3]
    for i in range(imax):
         for j in range(jmax):
              for k in range(kmax):
                ovector[i,j,k]=np.sum(weight[i] * ivector[:,sj*j:sj*j+pj,sk*k:sk*k+pk])
    return ovector

