import numpy as np
from Layer import Layer

class FullyConnectedLayer(Layer):
    def __init__(self, _length, _lr = 0.1):
        super(FullyConnectedLayer, self).__init__(_length, _lr)
        self._length = _length
        return

    def getLength(self):
        return self._length

    def append(self, _child):
        self._child = _child
        self._child.setParent(self)

        self._initWeights((self._length, self._child.getLength()))
        return self._child

    def type(self):
        return 'fullyconnected'
