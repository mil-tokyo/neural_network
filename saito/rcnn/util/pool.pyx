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
cdef unsigned int argmax2d(np.ndarray[DTYPE_t, ndim=2] node, unsigned int x_start, unsigned int x_span, unsigned int y_start, unsigned int y_span):
#def argmax2d(np.ndarray[DTYPE_t, ndim=2] node, unsigned int x_start, unsigned int x_span, unsigned int y_start, unsigned int y_span):
    cdef unsigned int i,j
    cdef unsigned int argmax = 0
    cdef float max_num = node[x_start,y_start]

    for i in range(x_start, x_start+x_span):
        for j in range(y_start, y_start+y_span):
            if max_num < node[i,j]:
                max_num = node[i,j]
                argmax = (i-x_start)*y_span+j-y_start
    return argmax

@cython.boundscheck(False)
def pool2d(np.ndarray[DTYPE_t, ndim=2] node, unsigned int kernel_x, unsigned int kernel_y, unsigned int stride_x, unsigned int stride_y):
    cdef unsigned int vertical = (node.shape[0] - kernel_x) / stride_x + 1
    cdef unsigned int horizontal = (node.shape[1] - kernel_y) / stride_y + 1
    cdef np.ndarray[DTYPE_t, ndim = 2] max_value = np.empty((vertical, horizontal))
    cdef np.ndarray[DTYPE_t, ndim = 2] pos = np.zeros((node.shape[0],node.shape[1]))
    cdef unsigned int i, j
    cdef unsigned int argmax = 0

    #for i in prange(vertical, nogil=True):
    for i in xrange(vertical):
        for j in xrange(horizontal):
            argmax = np.argmax(node[i*stride_x:i*stride_x+kernel_x,j*stride_y:j*stride_y+kernel_y])
            pos[i*stride_x+argmax/kernel_y, j*stride_y+argmax%kernel_y] = 1
            max_value[i,j] = node[i*stride_x+argmax/kernel_y, j*stride_y+argmax%kernel_y]
    return max_value, pos

@cython.boundscheck(False)
def pool3d(np.ndarray[DTYPE_t, ndim=3] node, int kernel_x, int kernel_y, int stride_x, int stride_y):
    cdef unsigned int vertical = (node.shape[1] - kernel_x) / stride_x + 1
    cdef unsigned int horizontal = (node.shape[2] - kernel_y) / stride_y + 1
    cdef np.ndarray[DTYPE_t, ndim = 3] max_value = np.empty((node.shape[0],vertical, horizontal))
    cdef np.ndarray[DTYPE_t, ndim = 3] pos = np.zeros((node.shape[0],node.shape[1],node.shape[2]))
    cdef unsigned int i, j, k
    cdef unsigned int argmax = 0

    #for i in prange(vertical, nogil=True):
    for k in xrange(node.shape[0]):
        for i in xrange(vertical):
            for j in xrange(horizontal):
                #argmax = np.argmax(node[k,i*stride_x:i*stride_x+kernel_x,j*stride_y:j*stride_y+kernel_y])
                argmax = argmax2d(node[k],i*stride_x,kernel_x,j*stride_y,kernel_y)
                pos[k, i*stride_x+argmax/kernel_y, j*stride_y+argmax%kernel_y] = 1
                max_value[k,i,j] = node[k, i*stride_x+argmax/kernel_y, j*stride_y+argmax%kernel_y]
                
    return max_value, pos

@cython.boundscheck(False)
def pool4d(np.ndarray[DTYPE_t, ndim=4] node, unsigned int kernel_x, unsigned int kernel_y, unsigned int kernel_z, unsigned int stride_x, unsigned int stride_y, unsigned int stride_z):
    cdef unsigned int vertical = (node.shape[1] - kernel_x) / stride_x + 1
    cdef unsigned int horizontal = (node.shape[2] - kernel_y) / stride_y + 1
    cdef unsigned int depth = (node.shape[0] - kernel_z) / stride_z + 1

    cdef unsigned int i,j,k,x,y,z,kernel_index
    cdef unsigned int argmax = 0
    cdef np.ndarray[DTYPE_t, ndim = 4] max_value = np.zeros((depth, vertical, horizontal, node.shape[3]))
    cdef np.ndarray[DTYPE_t, ndim = 4] pos = np.zeros((node.shape[0],node.shape[1],node.shape[2],node.shape[3]))

    cdef unsigned int tmp_x, tmp_y, tmp_z
    cdef float max_num


    for kernel_index in prange(node.shape[3], nogil=True):
    #for kernel_index in xrange(node.shape[3]):
        for x in xrange(vertical):
            for y in xrange(horizontal):
                for z in xrange(depth):
                    argmax = 0
                    max_num = node[z*stride_z, x*stride_x, y*stride_y, kernel_index]
                    for i in range(x*stride_x, x*stride_x+kernel_x):
                        for j in range(y*stride_y, y*stride_y+kernel_y):
                            for k in range(z*stride_z, z*stride_z+kernel_z):
                                if max_num < node[k,i,j,kernel_index]:
                                    max_num = node[k,i,j,kernel_index]
                                    argmax = (i-x*stride_x)*kernel_y*kernel_z+(j-y*stride_y)*kernel_z+(k-z*stride_z)
                    #argmax = argmax3d(node, x*stride_x, kernel_x, y*stride_y, kernel_y, z*stride_z, kernel_z, kernel_index)
                    tmp_x = argmax/(kernel_y * kernel_z)
                    tmp_y = (argmax - tmp_x * kernel_y * kernel_z) / kernel_z
                    tmp_z = (argmax - tmp_x * kernel_y * kernel_z - tmp_y * kernel_z)
                    pos[z*stride_z+tmp_z,x*stride_x+tmp_x,y*stride_y+tmp_y, kernel_index] += 1

                    max_value[z,x,y,kernel_index] = max_num
                    #max_value[z,x,y,kernel_index] = node[z*stride_z+tmp_z,x*stride_x+tmp_x,y*stride_y+tmp_y, kernel_index]
                
    return max_value, pos

@cython.boundscheck(False)
cdef unsigned int argmax3d(np.ndarray[DTYPE_t, ndim=4] node, unsigned int x_start, unsigned int x_span, unsigned int y_start, unsigned int y_span, unsigned int z_start, unsigned int z_span, unsigned int kernel_index):
#cdef argmax3d(np.ndarray[DTYPE_t, ndim=4] node, unsigned int x_start, unsigned int x_span, unsigned int y_start, unsigned int y_span, unsigned int z_start, unsigned int z_span, unsigned int kernel_index):
    cdef unsigned int i,j,k
    cdef unsigned int argmax = 0
    cdef float max_num = node[z_start,x_start,y_start,kernel_index]

    for i in range(x_start, x_start+x_span):
        for j in range(y_start, y_start+y_span):
            for k in range(z_start, z_start+z_span):
                if max_num < node[k,i,j,kernel_index]:
                    max_num = node[k,i,j,kernel_index]
                    argmax = (i-x_start)*y_span*z_span+(j-y_start)*z_span+(k-z_start)
    return argmax




# @cython.boundscheck(False)
# def spacial_pool(np.ndarray[DTYPE_t, ndim=4] node, unsigned int select_frame = 6):
#     cdef np.ndarray[DTYPE_t, ndim = 4] next_node = np.empty((select_frame, node.shape[1], node.shape[2],node.shape[3]))
#     cdef np.ndarray[DTYPE_t, ndim = 1] frame_count = np.empty(select_frame)

#     cdef unsigned int kernel, i, j, frame, time, avg
    
#     for kernel in xrange(node.shape[3]):
#         for i in xrange(node.shape[1]):
#             for j in xrange(node.shape[2]):
#                 for frame in xrange(select_frame):
#                     frame_count[frame] = node.shape[0] / select_frame * (frame+1) - node.shape[0] / select_frame * frame
#                     avg = 0
#                     for time in xrange(frame_count[frame]):
#                         avg = avg + self.node(self.node.)
                    
                    

#     cdef unsigned int vertical = (node.shape[1] - kernel_x) / stride_x + 1
#     cdef unsigned int horizontal = (node.shape[2] - kernel_y) / stride_y + 1
#     cdef unsigned int depth = (node.shape[0] - kernel_z) / stride_z + 1

#     cdef unsigned int i,j,k,x,y,z,kernel_index
#     cdef unsigned int argmax = 0
#     cdef np.ndarray[DTYPE_t, ndim = 4] max_value = np.zeros((depth, vertical, horizontal, node.shape[3]))
#     cdef np.ndarray[DTYPE_t, ndim = 4] pos = np.zeros((node.shape[0],node.shape[1],node.shape[2],node.shape[3]))

#     cdef unsigned int tmp_x, tmp_y, tmp_z
#     cdef float max_num


#     for kernel_index in prange(node.shape[3], nogil=True):
#     #for kernel_index in xrange(node.shape[3]):
#         for x in xrange(vertical):
#             for y in xrange(horizontal):
#                 for z in xrange(depth):
#                     argmax = 0
#                     max_num = node[z*stride_z, x*stride_x, y*stride_y, kernel_index]
#                     for i in range(x*stride_x, x*stride_x+kernel_x):
#                         for j in range(y*stride_y, y*stride_y+kernel_y):
#                             for k in range(z*stride_z, z*stride_z+kernel_z):
#                                 if max_num < node[k,i,j,kernel_index]:
#                                     max_num = node[k,i,j,kernel_index]
#                                     argmax = (i-x*stride_x)*kernel_y*kernel_z+(j-y*stride_y)*kernel_z+(k-z*stride_z)
#                     #argmax = argmax3d(node, x*stride_x, kernel_x, y*stride_y, kernel_y, z*stride_z, kernel_z, kernel_index)
#                     tmp_x = argmax/(kernel_y * kernel_z)
#                     tmp_y = (argmax - tmp_x * kernel_y * kernel_z) / kernel_z
#                     tmp_z = (argmax - tmp_x * kernel_y * kernel_z - tmp_y * kernel_z)
#                     pos[z*stride_z+tmp_z,x*stride_x+tmp_x,y*stride_y+tmp_y, kernel_index] += 1

#                     max_value[z,x,y,kernel_index] = max_num
#                     #max_value[z,x,y,kernel_index] = node[z*stride_z+tmp_z,x*stride_x+tmp_x,y*stride_y+tmp_y, kernel_index]
                
#     return max_value, pos

