import numpy as np

class Neuron:
    def __init__(self, map_size):
        self.weight = np.zeros((map_size,map_size))
        self.weight.astype(np.float)
