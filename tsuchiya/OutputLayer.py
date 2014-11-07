import numpy as np
from FullyConnectedLayer import FullyConnectedLayer

class OutputLayer(FullyConnectedLayer):
    def backward(self):
        self._calcDiff()
        if (not self._parent == None):
            self._parent.backward(self._diff)
        return

    def _calcDiff(self):
        self._diff = self.activate(self._units) - self._train_data

    def setTrainData(self, _train_data):
        self._train_data = _train_data




