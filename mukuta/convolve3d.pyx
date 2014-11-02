from __future__ import division
import numpy as np
cimport numpy as np
from cython.parallel import prange

def convolve3d(np.ndarray[np.float_t,ndim=3] ivector,np.ndarray[np.float_t,ndim=4] weight, np.int_t imax,np.int_t jmax,np.int_t kmax,np.int_t sj,np.int_t sk):
    cdef int i,j,k,l,pj,pk
    cdef np.ndarray[np.float_t,ndim=3] ovector=np.zeros((imax,jmax,kmax))
    cdef int pjmax=weight.shape[2]
    cdef int pkmax=weight.shape[3]
    cdef int lmax=ivector.shape[0]
    for i in prange(imax,nogil=True):
        for l in range(lmax):
           for j in range(jmax):
               for k in range(kmax):
                   for pj in range(pjmax):
                       for pk in range(pkmax):
                           ovector[i,j,k]+=weight[i,l,pj,pk] * ivector[l,sj*j+pj,sk*k+pk]
    return ovector

