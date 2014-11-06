import network as network
import numpy as np

num_unit_of_each_layer = np.array([784, 128, 10])
nn = network.neural_network.NeuralNetwork(num_unit_of_each_layer, 10)
nn.learn(epoch = 180000)
#nn.test()

