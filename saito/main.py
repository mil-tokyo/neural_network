import network as network
import numpy as np

#num_unit_of_each_layer = np.array([784, 128, 10])
#nn = network.neural_network.NeuralNetwork(num_unit_of_each_layer)
#nn.learn(epoch = 30)
#nn.test()

cnn = network.convolutional_neural_network.ConvolutionalNeuralNetwork()
cnn.learn(epoch = 30)
