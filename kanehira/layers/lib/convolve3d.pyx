import numpy as np
cimport numpy as np
ctypedef np.float_t DTYPE_t
cimport cython

def convolve3d(np.ndarray[DTYPE_t, ndim=3] input, np.ndarray[DTYPE_t, ndim=4] weight, unsigned int window_size, char* mode, unsigned int axis):
    cdef np.ndarray[DTYPE_t, ndim=3] filled_input
    cdef np.ndarray[DTYPE_t, ndim=3] output
    if mode[:] == "full":
        filled_input = pad0(input, window_size) 
        output = convolve3d_valid(filled_input, weight, window_size, axis)
        return output 
    elif mode[:] == "valid":
        output = convolve3d_valid(input, weight, window_size, axis)
        return output

@cython.boundscheck(False)
cdef np.ndarray[DTYPE_t, ndim=3] pad0(np.ndarray[DTYPE_t, ndim=3] input, unsigned int window_size):
    cdef unsigned int kernel_size = input.shape[0]
    cdef unsigned int input_size_row = input.shape[1]
    cdef unsigned int input_size_col = input.shape[2]
    cdef unsigned int output_size_row = input_size_row + 2 * (window_size - 1)
    cdef unsigned int output_size_col = input_size_col + 2 * (window_size - 1)
    cdef np.ndarray[DTYPE_t, ndim=3] output = np.zeros([kernel_size, output_size_row, output_size_col], dtype=np.float)
    cdef unsigned int i, j, k

    for k in range(kernel_size):
        for i in range(input_size_row):
            for j in range(input_size_col):
                output[k, i + (window_size - 1), j + (window_size - 1)] = input[k, i, j]
    return output
                

@cython.boundscheck(False)
cdef np.ndarray[DTYPE_t, ndim=3] convolve3d_valid(np.ndarray[DTYPE_t, ndim=3] input, np.ndarray[DTYPE_t, ndim=4] weight, unsigned int window_size, unsigned int axis):
    cdef unsigned int input_kernel_size
    cdef unsigned int output_kernel_size

    if axis == 0:
        input_kernel_size = weight.shape[1]
        output_kernel_size = weight.shape[0]
    elif axis == 1:
        input_kernel_size = weight.shape[0]
        output_kernel_size = weight.shape[1]

    cdef unsigned int input_size_row = input.shape[1]
    cdef unsigned int input_size_col = input.shape[2]
    cdef unsigned int output_size_row = input_size_row - window_size + 1
    cdef unsigned int output_size_col = input_size_col - window_size + 1

    cdef np.ndarray[DTYPE_t, ndim=3] output = np.zeros([output_kernel_size, output_size_row, output_size_col], dtype=np.float)
    cdef DTYPE_t value
    cdef unsigned int i, j, x, y, dx, dy

    if axis == 0:
        for j in range(output_kernel_size):
            for i in range(input_kernel_size):
                for y in range(output_size_row):
                    for x in range(output_size_col):
                        value = 0
                        for dy in range(window_size):
                            for dx in range(window_size):
                                value += input[i, y+dy, x+dx] * weight[j, i, dy, dx]
                        output[j, y, x] += value


    elif axis == 1:
        for j in range(output_kernel_size):
            for i in range(input_kernel_size):
                for y in range(output_size_row):
                    for x in range(output_size_col):
                        value = 0
                        for dy in range(window_size):
                            for dx in range(window_size):
                                value += input[i, y+dy, x+dx] * weight[i, j, dy, dx]
                        output[j, y, x] += value

    return output

@cython.boundscheck(False)
def convolve2d_with3d(np.ndarray[DTYPE_t, ndim=3] input, np.ndarray[DTYPE_t, ndim=3] weight):
    """ valid """
    cdef unsigned int input_kernel_size = input.shape[0]
    cdef unsigned int output_kernel_size = weight.shape[0]
    cdef unsigned int input_size_row = input.shape[1]
    cdef unsigned int input_size_col = input.shape[2]
    cdef unsigned int window_size = weight.shape[2]
    cdef unsigned int output_size_row = input_size_row - window_size + 1
    cdef unsigned int output_size_col = input_size_col - window_size + 1
    cdef np.ndarray[DTYPE_t, ndim=4] output = np.zeros([output_kernel_size, input_kernel_size, output_size_row, output_size_col], dtype=np.float)
    
    cdef unsigned int i, j, x, y, dx, dy
    cdef DTYPE_t value

    for j in range(output_kernel_size):
        for i in range(input_kernel_size):
            for y in range(output_size_row):
                for x in range(output_size_col):
                    value = 0
                    for dy in range(window_size):
                        for dx in range(window_size):
                            value += input[i, y+dy, x+dx] * weight[j, dy, dx]
                    output[j, i, y, x] = value

    return output
            




