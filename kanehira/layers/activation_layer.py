import numpy as np
from abstract_layer import Layer

class Activation(Layer):
    def __init__(self, layer_setting):
        activation_type = layer_setting["activation_type"]
        self.act_func = getattr(self, "_"+activation_type)
        self.d_act_func = getattr(self, "_d_"+activation_type)

    def forward_calculate(self, inp):
        self.output = self.act_func(inp)
        return self.output

    def back_calculate(self, prev_delta):
        return self.d_act_func(self.output) * prev_delta

    def update(self, eta, batch_size):
        pass

    def _tanh(self, x):
        return np.tanh(x)

    def _d_tanh(self, x):
        return 1 - np.power(x, 2) 

    def _sigmoid(self, x):
        return 1 / 1 + np.exp(-x)

    def _d_sigmoid(self, x):
        return x * (1-x)

    def _softmax(self, x):
        y = x - max(x) 
        return np.exp(y) / np.sum(np.exp(y))

    def _d_softmax(self, x):
        return 1

    def _ReLU(self, x):
        return np.maximum(x, 0)

    def _d_ReLU(self, x):
        return np.sign(x)

    def __str__(self):
        return "ReLU layer:"

