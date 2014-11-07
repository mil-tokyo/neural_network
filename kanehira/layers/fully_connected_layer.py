import numpy as np

class FCLayer:
    def __init__(self, layer_setting):
        self.input_shape = layer_setting["input_num"]
        self.output_num = layer_setting["output_num"]

        if isinstance(self.input_shape, tuple):
            self.is_reshape = True
            self.input_num = reduce(lambda a, b: a*b, self.input_shape)
        else:
            self.is_reshape = False
            self.input_num = self.input_shape

        self.W = np.random.uniform(-1, 1, size = (self.output_num, self.input_num) )
        self.inp = None

    def forward_calculate(self, inp):
        if self.is_reshape:
            self.inp = inp.flatten()
        else:
            self.inp = inp

        self.a = np.dot(self.W, self.inp)
        return self.a

    def back_calculate(self, prev_delta):
        if np.isnan(prev_delta).any():
            raise ValueError("nan value appeared in prev_delta at FCLayer backcalculatioin\n" + \
                                 "delta = {}".format(prev_delta))
        self.delta = prev_delta

        delta = np.dot(self.W.T, prev_delta)
        if self.is_reshape:
            return np.reshape(delta, self.input_shape)
        else: 
            return delta

    def update(self, eta):
        self.div = np.dot(np.matrix(self.delta).T, np.matrix(self.inp))
        self.W = np.array(self.W - eta * self.div)

        if np.isnan(self.W).any():
            raise ValueError("nan value appeared in weight matrix at FCLayer\n" + \
                                 "div = {}".format(self.div))

    def __str__(self):
        return "FCLayer W:{}".format(self.W)
