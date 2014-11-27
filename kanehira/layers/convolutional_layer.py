import numpy as np
from scipy import signal
from lib.convolve3d import convolve3d, convolve2d_with3d

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
        output = convolve3d(inp, self.W, self.window_size, mode='valid', axis=0)
        return output

    def back_calculate(self, prev_delta_map):
        """calculate back propagation"""
        self.delta_map = prev_delta_map
        delta_map = convolve3d(prev_delta_map, self.W[:, :, ::-1, ::-1], self.window_size, mode='full', axis=1)
        self.diff_W += convolve2d_with3d(self.input, self.delta_map)
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
    c = np.arange(27.0).reshape(3,3,3)
    b = np.ones((2,2,2))
    b[1:,:,:] =0
    print "foraward input",c
    print a.forward_calculate(c)

    print "bak input", b
    print a.back_calculate(b)
    print a.update(0.01)
