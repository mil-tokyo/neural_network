import numpy as np
cimport numpy as np
ctypedef np.float_t DTYPE_t
cimport cython

@cython.boundscheck(False)
def maxpooling(np.ndarray[DTYPE_t, ndim=3] input, unsigned int window_size, unsigned int stride):
    """ assuming window will not overlap """
    cdef unsigned int input_kernel_size = input.shape[0]
    cdef unsigned int input_row = input.shape[1]
    cdef unsigned int input_col = input.shape[2]
    cdef unsigned int output_kernel_size = input_kernel_size
    cdef unsigned int output_row = input_row / window_size
    cdef unsigned int output_col = input_col / window_size
    
    cdef np.ndarray[DTYPE_t, ndim=3] output = np.zeros([output_kernel_size, output_row, output_col], dtype=np.float)
    cdef np.ndarray[np.int_t, ndim=3] max_index_map = np.zeros([input_kernel_size, input_row, input_col], dtype=np.int)
    cdef unsigned int k, y, x, dy, dx
    cdef unsigned int max_index_x, max_index_y
    cdef float val, max_val

    for k in range(output_kernel_size):
        for y in range(output_row):
            for x in range(output_col):
                max_index_y = 0
                max_index_x = 0
                max_val = -10000
                for dy in range(window_size):
                    for dx in range(window_size):
                        val = input[k , y * stride + dy , x * stride + dx]
                        if val > max_val:
                            max_val = val
                            max_index_y = dy
                            max_index_x = dx
                output[k, y, x] = max_val
                max_index_map[k, y * stride + max_index_y, x * stride + max_index_x] = 1
            

    return output, max_index_map
