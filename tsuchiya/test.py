import numpy as np
from FullyConnectedLayer import FullyConnectedLayer
from OutputLayer import OutputLayer

l1 = FullyConnectedLayer(17*17)
l2 = FullyConnectedLayer(50)
l3 = OutputLayer(10)

l1.append(l2).append(l3)

l1.forward(np.ones(17*17))

print l1.getUnits()
print l2.getUnits()
print l3.getUnits()

l3.setTrainData(np.array([1,0,0,0,0,0,0,0,0,0]))
l3.backward()
