import numpy as np
class Activation():
    def __init__(self, layer_setting):
        pass

    def forward_calculate(self, inp):
        self.output = self._ReLU(inp)
        return self.output

    def back_calculate(self, prev_delta):
        return np.sign(self.output) * prev_delta

    def update(self, eta):
        pass

    def _ReLU(self, x):
        return np.maximum(x, 0)

    def _d_ReLU(self, x):
        return np.sign(x)

