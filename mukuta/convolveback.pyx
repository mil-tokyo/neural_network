from __future__ import division
import numpy as np
cimport numpy as np
import cython.parallel

def convolveback(np.ndarray[np.float_t,ndim=3] odiff,np.ndarray[np.float_t,ndim=4] weight, np.ndarray[np.float_t,ndim=3] ivector,np.int_t imax,np.int_t jmax,np.int_t kmax,np.int_t sj,np.int_t sk):

    cdef int i,j,k,l,pj,pk
    cdef int pjmax=weight.shape[2]
    cdef int pkmax=weight.shape[3]
    cdef int lmax=ivector.shape[0]
    cdef int ij=ivector.shape[1]
    cdef int ik=ivector.shape[2]
    cdef np.ndarray[np.float_t,ndim=4] dweight=np.zeros((imax,lmax,pjmax,pkmax))
    cdef np.ndarray[np.float_t,ndim=3] idiff=np.zeros((lmax,ij,ik))
    
    for i in cython.parallel.prange(imax,nogil=True):
        for l in range(lmax):
            for j in range(jmax):
                for k in range(kmax):
                    for pj in range(pjmax):
                        for pk in range(pkmax):
                            dweight[i,l,pj,pk]+=odiff[i,j,k]*ivector[l,sj*j+pj,sk*k+pk]
                            idiff[l,sj*j+pj,sk*k+pk]+=dweight[i,l,pj,pk]*odiff[i,j,k]
    return dweight,idiff

