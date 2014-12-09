import numpy as np

def convolve2d(a, b, mode = 'valid', padding = None):
    '''
    padding : taple
    a : np.array
    b : np.array (dimension of b is same to a)
    mode : 'valid' or 'same' or 'full'
    '''
    # b's size is restricted to odd number when using 'same' mode
    if padding is None:
        if mode == 'valid':
            padding = [0] * len(b.shape)
        if mode == 'same':
            padding = [0] * len(b.shape)
            for i in range(len(b.shape)):
                padding[i] = (b.shape[i] - 1) / 2
        if mode == 'full':
            padding = [0] * len(b.shape)
            for i in range(len(b.shape)):
                padding[i] = b.shape[i] - 1

    output_shape = [0] * len(b.shape)
    for i in range(len(output_shape)):
        output_shape[i] = a.shape[i] + padding[i] * 2 - b.shape[i] + 1

    ret = np.zeros(output_shape)
    for i in xrange(output_shape[0]):
        for j in xrange(output_shape[1]):
            for row in xrange(b.shape[0]):
                for col in xrange(b.shape[1]):
                    if i+row-padding[0] < 0 or i+row-padding[0] > a.shape[0] - 1 or j+col-padding[1]<0 or j+col-padding[1] > a.shape[1] - 1:
                        continue
                    ret[i,j] += a[i+row-padding[0],j+col-padding[1]] * b[row,col]
    return ret
        





