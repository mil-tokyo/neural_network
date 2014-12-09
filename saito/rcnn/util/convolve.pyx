import numpy as np
cimport numpy as np
from cython.parallel import prange
cimport cython

cdef extern from "stdlib.h":
    ctypedef unsigned int size_t
    size_t strlen(char *s)
    void * malloc(size_t size)
    void free(void *ptr)
    int strcmp(char *a, char *b)


DTYPE=np.float64
ctypedef np.float_t DTYPE_t

@cython.boundscheck(False)
def convolve2d(np.ndarray[DTYPE_t, ndim=2] a, np.ndarray[DTYPE_t, ndim=2] b, char* mode = 'valid', unsigned int padding_x = 0, unsigned int padding_y = 0, unsigned int padding_flug = 0, unsigned int rotation = 0):
    if padding_flug == 0:
        if not strcmp(mode,'valid'):
            padding_x = 0
            padding_y = 0
        elif not strcmp(mode,'same'):
            padding_x = (b.shape[0] - 1) / 2
            padding_y = (b.shape[1] - 1) / 2
        elif not strcmp(mode,'full'):
            padding_x = b.shape[0] - 1
            padding_y = b.shape[1] - 1
        else:
            raise NameError('unsupported mode for convolve2d')

    cdef unsigned int out_x = a.shape[0] + padding_x * 2 - b.shape[0] + 1
    cdef unsigned int out_y = a.shape[1] + padding_y * 2 - b.shape[1] + 1
    cdef np.ndarray[DTYPE_t, ndim=2] ret = np.zeros((out_x, out_y))
    cdef unsigned int i, j, row, col

    for i in xrange(out_x):
        for j in xrange(out_y):
            for row in xrange(b.shape[0]):
                for col in xrange(b.shape[1]):
                    if i+row-padding_x<0 or i+row-padding_x>a.shape[0] - 1 or j+col-padding_y<0 or j+col-padding_y>a.shape[1]-1:
                        continue
                    ret[i,j] += a[i+row-padding_x,j+col-padding_y] * b[row + rotation*(b.shape[0]-1-2*row), col + rotation*(b.shape[1]-1-2*col)]

    return ret

@cython.boundscheck(False)
def convolve3d_forward(np.ndarray[DTYPE_t, ndim=3] a, np.ndarray[DTYPE_t, ndim=4] b, char* mode = 'valid', unsigned int padding_x = 0, unsigned int padding_y = 0, unsigned int padding_flug = 0, unsigned int rotation = 0):
    if padding_flug ==0:
        if not strcmp(mode,'valid'):
            padding_x = 0
            padding_y = 0
        elif not strcmp(mode,'same'):
            padding_x = (b.shape[2] - 1) / 2
            padding_y = (b.shape[3] - 1) / 2
        elif not strcmp(mode,'full'):
            padding_x = b.shape[2] - 1
            padding_y = b.shape[3] - 1
        else:
            raise NameError('unsupported mode for convolve3d')

    cdef unsigned int out_x = a.shape[1] + padding_x * 2 - b.shape[2] + 1
    cdef unsigned int out_y = a.shape[2] + padding_y * 2 - b.shape[3] + 1
    cdef np.ndarray[DTYPE_t, ndim=3] ret = np.zeros((b.shape[0], out_x, out_y))
    cdef unsigned int k, l, i, j, row, col

    for k in xrange(b.shape[0]):
        for l in xrange(a.shape[0]):
            for i in xrange(out_x):
                for j in xrange(out_y):
                    for row in xrange(b.shape[2]):
                        for col in xrange(b.shape[3]):
                            if i+row-padding_x<0 or i+row-padding_x>a.shape[1] - 1 or j+col-padding_y<0 or j+col-padding_y>a.shape[2]-1:
                                continue
                            ret[k,i,j] += a[l,i+row-padding_x,j+col-padding_y] * b[k,l,row + rotation*(b.shape[2]-1-2*row), col + rotation*(b.shape[3]-1-2*col)]

    return ret

@cython.boundscheck(False)
def convolve3d_derr(np.ndarray[DTYPE_t, ndim=3] a, np.ndarray[DTYPE_t, ndim=4] b, char* mode = 'valid', unsigned int padding_x = 0, unsigned int padding_y = 0, unsigned int padding_flug = 0, unsigned int rotation = 0):
    if padding_flug ==0:
        if not strcmp(mode,'valid'):
            padding_x = 0
            padding_y = 0
        elif not strcmp(mode,'same'):
            padding_x = (b.shape[2] - 1) / 2
            padding_y = (b.shape[3] - 1) / 2
        elif not strcmp(mode,'full'):
            padding_x = b.shape[2] - 1
            padding_y = b.shape[3] - 1
        else:
            raise NameError('unsupported mode for convolve3d')

    cdef unsigned int out_x = a.shape[1] + padding_x * 2 - b.shape[2] + 1
    cdef unsigned int out_y = a.shape[2] + padding_y * 2 - b.shape[3] + 1
    cdef np.ndarray[DTYPE_t, ndim=3] ret = np.zeros((b.shape[1], out_x, out_y))
    cdef unsigned int k, l, i, j, row, col

    for k in xrange(b.shape[0]):
        for l in xrange(b.shape[1]):
            for i in xrange(out_x):
                for j in xrange(out_y):
                    for row in xrange(b.shape[2]):
                        for col in xrange(b.shape[3]):
                            if i+row-padding_x<0 or i+row-padding_x>a.shape[1] - 1 or j+col-padding_y<0 or j+col-padding_y>a.shape[2]-1:
                                continue
                            ret[l,i,j] += a[k,i+row-padding_x,j+col-padding_y] * b[k,l,row + rotation*(b.shape[2]-1-2*row), col + rotation*(b.shape[3]-1-2*col)]

    return ret

@cython.boundscheck(False)
def convolve3d_dweight(np.ndarray[DTYPE_t, ndim=3] a, np.ndarray[DTYPE_t, ndim=3] b, char* mode = 'valid', unsigned int padding_x = 0, unsigned int padding_y = 0, unsigned int padding_flug = 0, unsigned int rotation = 0):
    if padding_flug ==0:
        if not strcmp(mode,'valid'):
            padding_x = 0
            padding_y = 0
        elif not strcmp(mode,'same'):
            padding_x = (b.shape[1] - 1) / 2
            padding_y = (b.shape[2] - 1) / 2
        elif not strcmp(mode,'full'):
            padding_x = b.shape[1] - 1
            padding_y = b.shape[2] - 1
        else:
            raise NameError('unsupported mode for convolve3d')

    cdef unsigned int out_x = a.shape[1] + padding_x * 2 - b.shape[1] + 1
    cdef unsigned int out_y = a.shape[2] + padding_y * 2 - b.shape[2] + 1
    cdef np.ndarray[DTYPE_t, ndim=4] ret = np.zeros((b.shape[0], a.shape[0], out_x, out_y))
    cdef unsigned int k, l, i, j, row, col

    for k in xrange(a.shape[0]):
        for l in xrange(b.shape[0]):
            for i in xrange(out_x):
                for j in xrange(out_y):
                    for row in xrange(b.shape[1]):
                        for col in xrange(b.shape[2]):
                            if i+row-padding_x<0 or i+row-padding_x>a.shape[1] - 1 or j+col-padding_y<0 or j+col-padding_y>a.shape[2]-1:
                                continue
                            ret[l,k,i,j] += a[k,i+row-padding_x,j+col-padding_y] * b[l,row + rotation*(b.shape[2]-1-2*row), col + rotation*(b.shape[3]-1-2*col)]

    return ret

@cython.boundscheck(False)
cdef np.ndarray[DTYPE_t, ndim = 3] activate3d(np.ndarray[DTYPE_t, ndim = 3] a):
    cdef unsigned int i,j,k
    for i in xrange(a.shape[0]):
        for j in xrange(a.shape[1]):
            for k in xrange(a.shape[2]):
                a[i,j,k] = 1.0/(1+np.exp(-a[i,j,k]))
    return a

@cython.boundscheck(False)
def convolve_recurrent(np.ndarray[DTYPE_t, ndim = 4] a, np.ndarray[DTYPE_t, ndim = 4] b):
    cdef unsigned int padding_x = (b.shape[2] - 1)/2, padding_y = (b.shape[3] - 1)/2
    cdef unsigned int i,j,k,l,m,row,col
    cdef np.ndarray[DTYPE_t, ndim=3] tmp = np.empty((a.shape[1],a.shape[2],a.shape[3]))
    ac = lambda x : 1.0 / (1+np.exp(-x))
    a[0] = ac(a[0])
    cdef int a_x, a_y
    cdef double napier = np.e

    for i in prange(a.shape[0] - 1, nogil = True):
    #for i in xrange(a.shape[0] - 1):
        for j in xrange(tmp.shape[0]):
            for k in xrange(tmp.shape[1]):
                for l in xrange(tmp.shape[2]):
                    tmp[j,k,l] = 0
        for j in xrange(b.shape[0]):
            for k in xrange(b.shape[1]):
                for l in xrange(a.shape[1]):
                    for m in xrange(a.shape[2]):
                        for row in xrange(b.shape[2]):
                            for col in xrange(b.shape[3]):
                                a_x = l+row-padding_x
                                a_y = m+col-padding_y
                                if a_x<0 or a_x>a.shape[1]-1 or a_y<0 or a_y>a.shape[2]-1:
                                    continue
                                tmp[l,m,j] += a[i,a_x,a_y,k] * b[j,k,row,col]
        for j in xrange(a.shape[1]):
            for k in xrange(a.shape[2]):
                for l in xrange(a.shape[3]):
                    a[i+1,j,k,l] = 1.0 / (1+napier**(-a[i+1,j,k,l]-tmp[j,k,l]))
    return a

@cython.boundscheck(False)
def convolve_recurrent_backward(np.ndarray[DTYPE_t, ndim = 4] a, np.ndarray[DTYPE_t, ndim = 4] b, unsigned int padding_x, unsigned int padding_y):

    cdef unsigned int i,j,k,l,m,row,col
    cdef unsigned int out_x = padding_x * 2 + 1, out_y = padding_y * 2 + 1
    cdef np.ndarray[DTYPE_t, ndim=4] ret = np.zeros((a.shape[3],a.shape[3],out_x, out_y))

    cdef int a_x, a_y

    for i in prange(a.shape[3], nogil=True):
        for j in xrange(a.shape[3]):
            for k in xrange(a.shape[0] - 1):
                for l in xrange(out_x):
                    for m in xrange(out_y):
                        for row in xrange(b.shape[1]):
                            for col in xrange(b.shape[2]):
                                a_x = l+row-padding_x
                                a_y = m+col-padding_y
                                if a_x<0 or a_x>a.shape[1]-1 or a_y<0 or a_y>a.shape[2]-1:
                                    continue
                                ret[i,j,l,m] += a[k,a_x,a_y,j] * b[k+1,row,col,i]
    return ret


@cython.boundscheck(False)
def convolve4d_forward(np.ndarray[DTYPE_t, ndim = 4] node, np.ndarray[DTYPE_t, ndim = 5] weight, char* mode = 'valid', unsigned int padding_x = 0, unsigned int padding_y = 0, unsigned int padding_z = 0, unsigned int padding_flug = 0, unsigned int rotation = 0):
    if padding_flug ==0:
        if not strcmp(mode,'valid'):
            padding_x = 0
            padding_y = 0
            padding_z = 0
        elif not strcmp(mode,'same'):
            padding_x = (weight.shape[2] - 1) / 2
            padding_y = (weight.shape[3] - 1) / 2
            padding_z = (weight.shape[4] - 1) / 2
        elif not strcmp(mode,'full'):
            padding_x = weight.shape[2] - 1
            padding_y = weight.shape[3] - 1
            padding_z = weight.shape[4] - 1
        else:
            raise NameError('unsupported mode for convolve3d')

    cdef unsigned int out_x = node.shape[1] + 2*padding_x - weight.shape[2] + 1
    cdef unsigned int out_y = node.shape[2] + 2*padding_y - weight.shape[3] + 1
    cdef unsigned int out_z = node.shape[0] + 2*padding_z - weight.shape[4] + 1
    cdef np.ndarray[DTYPE_t, ndim = 4] next_node = np.zeros((out_z, out_x, out_y, weight.shape[0]))
    
    cdef unsigned int i,j,x,y,z,row,col,depth
    cdef int node_x, node_y, node_z
    cdef unsigned int s
    for s in prange(weight.shape[0]*weight.shape[1], nogil=True):
        i = s / weight.shape[1]
        j = s % weight.shape[1]
        for x in xrange(next_node.shape[1]):
            for y in xrange(next_node.shape[2]):
                for z in xrange(next_node.shape[0]):
                    for row in xrange(weight.shape[2]):
                        for col in xrange(weight.shape[3]):
                            for depth in xrange(weight.shape[4]):
                                node_x = x+row-padding_x
                                node_y = y+col-padding_y
                                node_z = z+depth - padding_z
                                if node_x<0 or node_x>node.shape[1]-1 or node_y<0 or node_y>node.shape[2]-1 or node_z<0 or node_z>node.shape[0]:
                                    continue
                                next_node[z,x,y,i] += node[node_z, node_x, node_y, j] * weight[i,j,row+rotation*(weight.shape[2]-1-2*row),col+rotation*(weight.shape[3]-1-2*col),depth+rotation*(weight.shape[4]-1-2*depth)]

    # for i in prange(weight.shape[0], nogil=True):
    #     for j in xrange(weight.shape[1]):
    #         for x in xrange(next_node.shape[1]):
    #             for y in xrange(next_node.shape[2]):
    #                 for z in xrange(next_node.shape[0]):
    #                     for row in xrange(weight.shape[2]):
    #                         for col in xrange(weight.shape[3]):
    #                             for depth in xrange(weight.shape[4]):
    #                                 node_x = x+row-padding_x
    #                                 node_y = y+col-padding_y
    #                                 node_z = z+depth - padding_z
    #                                 if node_x<0 or node_x>node.shape[1]-1 or node_y<0 or node_y>node.shape[2]-1 or node_z<0 or node_z>node.shape[0]:
    #                                     continue
    #                                 next_node[z,x,y,i] += node[node_z, node_x, node_y, j] * weight[i,j,row+rotation*(weight.shape[2]-1-2*row),col+rotation*(weight.shape[3]-1-2*col),depth+rotation*(weight.shape[4]-1-2*depth)]

    return next_node

@cython.boundscheck(False)
def convolve4d_backward(np.ndarray[DTYPE_t, ndim = 4] next_derr, np.ndarray[DTYPE_t, ndim = 5] weight, char* mode = 'valid', unsigned int padding_x = 0, unsigned int padding_y = 0, unsigned int padding_z = 0, unsigned int padding_flug = 0, unsigned int rotation = 0):
    if padding_flug ==0:
        if not strcmp(mode,'valid'):
            padding_x = 0
            padding_y = 0
            padding_z = 0
        elif not strcmp(mode,'same'):
            padding_x = (weight.shape[2] - 1) / 2
            padding_y = (weight.shape[3] - 1) / 2
            padding_z = (weight.shape[4] - 1) / 2
        elif not strcmp(mode,'full'):
            padding_x = weight.shape[2] - 1
            padding_y = weight.shape[3] - 1
            padding_z = weight.shape[4] - 1
        else:
            raise NameError('unsupported mode for convolve3d')

    cdef unsigned int out_x = next_derr.shape[1] + 2*padding_x - weight.shape[2] + 1
    cdef unsigned int out_y = next_derr.shape[2] + 2*padding_y - weight.shape[3] + 1
    cdef unsigned int out_z = next_derr.shape[0] + 2*padding_z - weight.shape[4] + 1
    cdef np.ndarray[DTYPE_t, ndim = 4] derr = np.zeros((out_z, out_x, out_y, weight.shape[1]))
    
    cdef unsigned int i,j,x,y,z,row,col,depth
    cdef int node_x, node_y, node_z
    cdef unsigned int s
    for s in prange(weight.shape[0]*weight.shape[1], nogil=True):
        i = s / weight.shape[1]
        j = s % weight.shape[1]
        for x in xrange(derr.shape[1]):
            for y in xrange(derr.shape[2]):
                for z in xrange(derr.shape[0]):
                    for row in xrange(weight.shape[2]):
                        for col in xrange(weight.shape[3]):
                            for depth in xrange(weight.shape[4]):
                                node_x = x+row-padding_x
                                node_y = y+col-padding_y
                                node_z = z+depth - padding_z
                                if node_x<0 or node_x>next_derr.shape[1]-1 or node_y<0 or node_y>next_derr.shape[2]-1 or node_z<0 or node_z>next_derr.shape[0]:
                                    continue
                                derr[z,x,y,j] += next_derr[node_z, node_x, node_y, i] * weight[i,j,row+rotation*(weight.shape[2]-1-2*row),col+rotation*(weight.shape[3]-1-2*col),depth+rotation*(weight.shape[4]-1-2*depth)]

    return derr




@cython.boundscheck(False)
def convolve4d_dweight(np.ndarray[DTYPE_t, ndim = 4] node, np.ndarray[DTYPE_t, ndim = 4] next_derr, char* mode = 'valid', unsigned int padding_x = 0, unsigned int padding_y = 0, unsigned int padding_z = 0, unsigned int padding_flug = 0, unsigned int rotation = 0):
    #node, next_derr : time, x, y, kernel
    #weight : output_kernel, input_kernel, x, y, time
    if padding_flug ==0:
        if not strcmp(mode,'valid'):
            padding_x = 0
            padding_y = 0
            padding_z = 0
        elif not strcmp(mode,'same'):
            padding_x = (next_derr.shape[1] - 1) / 2
            padding_y = (next_derr.shape[2] - 1) / 2
            padding_z = (next_derr.shape[0] - 1) / 2
        elif not strcmp(mode,'full'):
            padding_x = next_derr.shape[1] - 1
            padding_y = next_derr.shape[2] - 1
            padding_z = next_derr.shape[0] - 1
        else:
            raise NameError('unsupported mode for convolve3d')

    cdef unsigned int out_x = node.shape[1] + 2*padding_x - next_derr.shape[1] + 1
    cdef unsigned int out_y = node.shape[2] + 2*padding_y - next_derr.shape[2] + 1
    cdef unsigned int out_z = node.shape[0] + 2*padding_z - next_derr.shape[0] + 1

    cdef np.ndarray[DTYPE_t, ndim = 5] dweight = np.zeros((next_derr.shape[3],node.shape[3],out_x, out_y, out_z))

    cdef unsigned int i,j,x,y,z,row,col,depth,s
    cdef int node_x, node_y, node_z
    cdef double tmp

    for s in prange(dweight.shape[0]*dweight.shape[1],nogil=True):
        i = s / dweight.shape[1]
        j = s % dweight.shape[1]
        for x in xrange(dweight.shape[2]):
            for y in xrange(dweight.shape[3]):
                for z in xrange(dweight.shape[4]):
                    for row in xrange(next_derr.shape[1]):
                        for col in xrange(next_derr.shape[2]):
                            for depth in xrange(next_derr.shape[0]):
                                node_x = x+row-padding_x
                                node_y = y+col-padding_y
                                node_z = z+depth-padding_z
                                if node_x<0 or node_x>node.shape[1]-1 or node_y<0 or node_y>node.shape[2]-1 or node_z<0 or node_z>node.shape[0] - 1:
                                    continue
                                dweight[i,j,x,y,z] += node[node_z, node_x, node_y, j] * next_derr[depth+rotation*(next_derr.shape[0]-1-2*depth),row+rotation*(next_derr.shape[1]-1-2*row),col+rotation*(next_derr.shape[2]-1-2*col),i]

    # for s in prange(dweight.shape[0]*dweight.shape[1]*dweight.shape[2]*dweight.shape[3]*dweight.shape[4]*next_derr.shape[1]*next_derr.shape[2]*next_derr.shape[0],nogil=True):
    #     index[0] = s
    #     i = index[0] / (dweight.shape[1]*dweight.shape[2]*dweight.shape[3]*dweight.shape[4]*next_derr.shape[1]*next_derr.shape[2]*next_derr.shape[0])
    #     index[1] = index[0] % (dweight.shape[1]*dweight.shape[2]*dweight.shape[3]*dweight.shape[4]*next_derr.shape[1]*next_derr.shape[2]*next_derr.shape[0])
    #     j = index[1] / (dweight.shape[2]*dweight.shape[3]*dweight.shape[4]*next_derr.shape[1]*next_derr.shape[2]*next_derr.shape[0])
    #     index[2] = index[1] % (dweight.shape[2]*dweight.shape[3]*dweight.shape[4]*next_derr.shape[1]*next_derr.shape[2]*next_derr.shape[0])
    #     x = index[2] / (dweight.shape[3]*dweight.shape[4]*next_derr.shape[1]*next_derr.shape[2]*next_derr.shape[0])
    #     index[3] = index[2] % (dweight.shape[3]*dweight.shape[4]*next_derr.shape[1]*next_derr.shape[2]*next_derr.shape[0])
    #     y = index[3] / (dweight.shape[4]*next_derr.shape[1]*next_derr.shape[2]*next_derr.shape[0])
    #     index[4] = index[3] % (dweight.shape[4]*next_derr.shape[1]*next_derr.shape[2]*next_derr.shape[0])
    #     z = index[4] / (next_derr.shape[1]*next_derr.shape[2]*next_derr.shape[0])
    #     index[5] = index[4] % (next_derr.shape[1]*next_derr.shape[2]*next_derr.shape[0])
    #     row = index[5] / (next_derr.shape[2]*next_derr.shape[0])
    #     index[6] = index[5] % (next_derr.shape[2]*next_derr.shape[0])
    #     col = index[6] / next_derr.shape[0]
    #     depth = index[6] % next_derr.shape[0]
    #     node_x = x+row-padding_x
    #     node_y = y+col-padding_y
    #     node_z = z+depth - padding_z
    #     if node_x<0 or node_x>node.shape[1]-1 or node_y<0 or node_y>node.shape[2]-1 or node_z<0 or node_z>node.shape[0] - 1:
    #         continue
    #     dweight[i,j,x,y,z] += node[node_z, node_x, node_y, j] * next_derr[depth+rotation*(next_derr.shape[0]-1-2*depth),row+rotation*(next_derr.shape[1]-1-2*row),col+rotation*(next_derr.shape[2]-1-2*col),i]

        
    # for i in prange(dweight.shape[0], nogil=True):
    #     for j in xrange(dweight.shape[1]):
    #         for x in xrange(dweight.shape[2]):
    #             for y in xrange(dweight.shape[3]):
    #                 for z in xrange(dweight.shape[4]):
    #                     for row in xrange(next_derr.shape[1]):
    #                         for col in xrange(next_derr.shape[2]):
    #                             for depth in xrange(next_derr.shape[0]):
    #                                 node_x = x+row-padding_x
    #                                 node_y = y+col-padding_y
    #                                 node_z = z+depth - padding_z
    #                                 if node_x<0 or node_x>node.shape[1]-1 or node_y<0 or node_y>node.shape[2]-1 or node_z<0 or node_z>node.shape[0] - 1:
    #                                     continue
    #                                 dweight[i,j,x,y,z] += node[node_z, node_x, node_y, j] * next_derr[depth+rotation*(next_derr.shape[0]-1-2*depth),row+rotation*(next_derr.shape[1]-1-2*row),col+rotation*(next_derr.shape[2]-1-2*col),i]
    return dweight




