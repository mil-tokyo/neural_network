from abstract_layer import Layer
import numpy as np
import time
from lib.maxpooling import maxpooling

class MaxPooling(Layer):
    def __init__(self, layer_setting):
        self.window_size = 2
        self.stride = 2

    def forward_calculate(self, inp):
        """not overlap"""
        output, self.max_index_map = maxpooling(inp, self.window_size, self.stride)
        return output

    def back_calculate(self, prev_delta):
        prev_kernel_size, prev_row, prev_col = prev_delta.shape
        rep_prev_delta = np.repeat(np.repeat(prev_delta, self.window_size, axis=1), self.window_size, axis=2)
        return rep_prev_delta * self.max_index_map

    def update(self, eta, batch_size):
        pass

    def __str__(self):
        return "MaxPooling Layer"


if __name__ == "__main__":
    a = {"a":"b"}
    mp = MaxPooling(a)
    b = np.arange(64.0).reshape(4,4,4)
    print b
    print mp.forward_calculate(b)
    delta = np.ones((4,2,2))
    print mp.back_calculate(delta)
