import numpy as np

class MaxPooling():
    def __init__(self, layer_setting):
        self.window_size = 2
        self.stride = 2

    def forward_calculate(self, inp):
        """not overlap"""
        input_kernel_size, input_row, input_col = inp.shape
        output_row = input_row / self.stride
        output_col = input_col / self.stride
        output = np.zeros((input_kernel_size, output_row, output_col))
        self.max_index_map = np.zeros(inp.shape)

        for y in xrange(output_row):
            for x in xrange(output_col):
                patch = inp[:, y * self.window_size : (y + 1) * self.window_size,\
                                   x * self.window_size : (x + 1) * self.window_size]
                output[:, x, y] = np.max(np.max(patch, axis = 1), axis = 1)
                for i in xrange(input_kernel_size):
                    sub_patch = patch[i, :, :]
                    max_index = np.where(sub_patch == np.max(sub_patch))
                    self.max_index_map[i,  y * self.window_size : (y + 1) * self.window_size,\
                                   x * self.window_size : (x + 1)* self.window_size ][ max_index ] = 1

        if np.isnan(output).any():
            ValueError("nan value appears in weight maxrix at MaxpoolingLayer forwardcalculation")

        return output

    def back_calculate(self, prev_delta):
        prev_kernel_size, prev_row, prev_col = prev_delta.shape
        rep_prev_delta = np.repeat(np.repeat(prev_delta, self.window_size, axis=1), self.window_size, axis=2)
        if np.isnan(prev_delta).any():
            raise ValueError("nan value appears in weight maxrix at MaxpoolingLayer backpropagation")
        return rep_prev_delta * self.max_index_map

    def update(self, eta):
        pass

    def __str__(self):
        return "MaxPooling"
