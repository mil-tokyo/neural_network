import numpy as np
from ConvolutionalLayer import ConvolutionalLayer
from PoolingLayer import PoolingLayer
from FullyConnectedLayer import FullyConnectedLayer
from OutputLayer import OutputLayer

lr = 0.01
l1 = ConvolutionalLayer((1, 28, 28), lr);
l2 = ConvolutionalLayer((10, 24, 24), lr, stride=1, window=(5, 5))
l2_pool = PoolingLayer((10, 12, 12), overlap=0, window=(2, 2))
l3 = ConvolutionalLayer((12, 10, 10), lr, stride=1, window=(3, 3))
l3_pool = PoolingLayer((12, 5, 5), overlap=0, window=(2, 2))
l4 = FullyConnectedLayer(128, lr)
l5 = OutputLayer(10, lr)

l1.append(l2).append(l2_pool).append(l3).append(l3_pool).append(l4).append(l5)

l1.forward(np.ones((1, 28, 28)))

l5.setTrainData(np.array([1,0,0,0,0,0,0,0,0,0]))
l5.backward()
