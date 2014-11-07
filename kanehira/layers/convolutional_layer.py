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
        self.W = np.ones((self.output_kernel_size, self.input_kernel_size, self.window_size, self.window_size))
#        self.W = np.random.uniform(-1,1,size=(self.output_kernel_size, self.input_kernel_size, self.window_size, self.window_size))

    def forward_calculate(self, inp):
        """calculate convolution process"""
        self.input = inp
        output = np.zeros((self.output_kernel_size, self.output_row, self.output_col))
        for j in xrange(self.output_kernel_size):
            output[j, :, :] = signal.convolve(inp, self.W[j, :, :, :], mode='valid')

        if np.isnan(output).any():
            raise ValueError("nan appears in output at ConvLayer forward")

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

        if np.isnan(self.delta_map).any():
            print self.delta_map
            raise ValueError("nan value appeared in weight matrix at ConvLayer, backcalculate\n"\
                                 "delta_map({}) =\n {}".format(self.delta_map.shape, self.delta_map))
        return delta_map

    def update(self, eta):
        self.diff_W = np.zeros(self.W.shape)
        for j in xrange(self.output_kernel_size):
            for y in xrange(self.output_row):
                for x in xrange(self.output_col):
                    self.diff_W[j, :, :, :] += self.input[:, y : y + self.window_size, x : x + self.window_size] * self.delta_map[j, y, x]

        print self.diff_W
        self.W -= eta * self.diff_W

        if np.isnan(self.W).any():
            raise ValueError("nan value appeared in weight matrix at ConvLayer" +\
                                 "diff_W =\n {}\ninput =\n{}\n delta(shape = {}) = \n{}"\
                                 .format(self.diff_W.shape, self.diff_W, self.input.shape,\
                                             self.input.shape, self.delta_map.shape, self.delta_map))
                        
    def __str__(self):
        return "ConvolutionalLayer W:{}".format(self.W)

if __name__ == "__main__":
    ns = {"input_kernel_size" : 3,
          "input_shape" : (3, 3),
          "output_kernel_size" : 1,
          "output_shape" : (2, 2),
          "window_size" : 2,
          "step_size" : 1
          }
      

    a = ConvolutionalLayer(ns)
    b = np.ones((1,2,2))
    c = np.ones((3,3,3))
    print a.forward_calculate(c)
    print a.back_calculate(b)

