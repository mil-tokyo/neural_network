import numpy as np
import scipy.signal

class Layer(object):
    _child = None
    _parent = None
    _units = None

    def __init__(self, _shape, _lr = 0.1):
        self._units = np.empty(_shape)
        #self.setActivationFunc(None)
        self.setActivationFunc(np.tanh)
        self.setActivationFuncDiff(Layer.tanhDiff)
        self._lr = _lr
        return

    def setParent(self, _parent):
        self._parent = _parent
        return

    def _initWeights(self, _shape):
        self._weights = np.random.normal(0, 0.05, size=_shape)
        self._bias = np.random.normal(0, 0.05, size=(_shape[1]))
        self._diff = np.empty(self._weights.shape)
        return


    def forward(self, _units):
        self._units = _units

        self._outputs = self.activate(self._units)

        if (not self._child == None):
            self._child.forward(np.dot(self._outputs, self._weights) + self._bias)

        return

    def backward(self, _prev_diff):
        self._update(_prev_diff)

        if (not self._parent == None):
            self._calcDiff(_prev_diff)
            self._parent.backward(self._diff)
        return

    def _update(self, _prev_diff):
        self._weights = self._weights - self._lr * np.outer(self._outputs, _prev_diff)
        #self._bias = self._bias - self._lr * _prev_diff_bias
        self._bias = self._bias - self._lr * _prev_diff
        return

    def _calcDiff(self, _prev_diff):
        self._diff = self._activation_func_diff(self._units) * np.dot(self._weights, _prev_diff)
        return

    def activate(self, _input):
        if self._activation_func == None:
            return _input
        else:
            return self._activation_func(_input)

    def setActivationFunc(self, _func):
        self._activation_func = _func
        return

    def setActivationFuncDiff(self, _func):
        self._activation_func_diff = _func


    def getUnits(self):
        return self._units

    @staticmethod
    def tanhDiff(_input):
        return 1.0 / np.square(np.cosh(_input))
    
    @staticmethod
    def sigmoid(_input, _alpha = 1.0):
        return 1.0 / (1.0 + np.exp(-_input*_alpha))

    @staticmethod
    def sigmoidDiff(_input, _alpha = 1.0):
        s = Layer.sigmoid(_input, _alpha)
        return s * (1-s)


    @staticmethod
    def convConvForward(weights, prev_units, next_shape):
        next_units = np.zeros(next_shape)
        for i in range(0, prev_units.shape[0]):
            next_units[i] = scipy.signal.convolve(prev_units, weights[:,i,:,:], 'valid')

        return next_units