import numpy as np
from scipy import signal

class ConvolutionalLayer():
    def __init__(self, layer_setting):
        self.input_row, self.input_col = layer_setting["input_shape"]
        self.input_kernel_size = layer_setting["input_kernel_size"]
        self.window_size = layer_setting["window_size"]
        self.step_size = layer_setting["step_size"]
        self.output_row, self.output_col = layer_setting["output_shape"]
        self.output_kernel_size = layer_setting["output_kernel_size"]

        
        if  self.output_row != self.input_row - self.window_size + 1 or\
                self.output_col != self.input_col - self.window_size + 1:
            true_shape = (self.input_row - self.window_size + 1,\
                                             self.input_col - self.window_size + 1 )
            raise ValueError("output shape should be {} but {}"\
                                 .format(true_shape, (self.output_row, self.output_col)))

        self.W = np.random.uniform(-1,1,size=(self.output_kernel_size, self.input_kernel_size, self.window_size, self.window_size))

    def forward_calculate(self, inp):
        """calculate convolution process"""
        self.input = inp
        output = np.zeros((self.output_kernel_size, self.output_row, self.output_col))
        for j in xrange(self.output_kernel_size):
            output[j, :, :] = signal.convolve(inp, self.W[j, :, :, :], mode='valid')

        if output.shape != (self.output_kernel_size, self.output_row, self.output_col):
            raise ValueError("the shape of matrix is wrong\n shape should be({}, {}, {}) but the shape of output is {}"\
                                 .format(self.output_kernel_size, self.output_row, self.output_col, output.shape))
        return output

    def back_calculate(self, prev_delta_map):
        """calculate back propagation"""
        self.delta_map = prev_delta_map
        conv_res = np.zeros((self.output_kernel_size, self.input_kernel_size, self.input_row, self.input_col))
        mirror_W = self.W[:, :, ::-1, ::-1]
        for i in xrange(self.input_kernel_size):
            for j in xrange(self.output_kernel_size):
                conv_res[j, i, :, :] = signal.convolve(prev_delta_map[j, :, :], mirror_W[j, i, :, :], mode='full')
        delta_map = np.sum(conv_res, axis = 0)
        return delta_map

    def update(self, eta):
        for i in xrange(self.input_kernel_size):
            for j in xrange(self.output_kernel_size):
                for y in xrange(self.input_row - self.window_size + 1):
                    for x in xrange(self.input_col - self.window_size + 1):
                        self.W[j, i, :, :] -= eta * self.input[i, y : y + self.window_size, x : x + self.window_size] * self.delta_map[j, y, x]

                        

