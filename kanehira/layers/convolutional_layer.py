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
        self.W = np.random.normal(loc=0, scale=0.1, size=(self.output_kernel_size, self.input_kernel_size, self.window_size, self.window_size))
        self.diff_W = np.zeros(self.W.shape)
        
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
        delta_map = np.zeros((self.input_kernel_size, self.input_row, self.input_col))

        for i in xrange(self.input_kernel_size):
            for j in xrange(self.output_kernel_size):
                delta_map[i, :, :] += signal.convolve(prev_delta_map[j, :, :], self.W[j, i, ::-1, ::-1], mode='full')

        for i in xrange(self.input_kernel_size):
            for j in xrange(self.output_kernel_size):
                self.diff_W[j, i, :, :] += signal.convolve(self.input[i, :, :], self.delta_map[j, ::-1, ::-1], mode = "valid")[::-1, ::-1]

        if np.isnan(self.delta_map).any():
            print self.delta_map
            raise ValueError("nan value appeared in weight matrix at ConvLayer, backcalculate\n"\
                                 "delta_map({}) =\n {}".format(self.delta_map.shape, self.delta_map))
        return delta_map

    def update(self, eta, batch_size):            
        self.W -= eta * self.diff_W / batch_size
        self.diff_W = np.zeros(self.W.shape)
                        
    def __str__(self):
        return "ConvolutionalLayer W:{}".format(self.W)

if __name__ == "__main__":
    ns = {"input_kernel_size" : 3,
          "input_shape" : (3, 3),
          "output_kernel_size" : 2,
          "output_shape" : (2, 2),
          "window_size" : 2,
          "step_size" : 1
          }
      
    a = ConvolutionalLayer(ns)
    c = np.arange(27).reshape(3,3,3)
    b = np.ones((2,2,2))
    b[1:,:,:] =0
    print "foraward input",c
    print a.forward_calculate(c)

    print "bak input", b
    print a.back_calculate(b)
    print a.update(0.01)
