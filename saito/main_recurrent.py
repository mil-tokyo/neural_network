import recurrent_network as network
import numpy as np

#num_unit_of_each_layer = np.array([784, 128, 10])
#nn = network.neural_network.NeuralNetwork(num_unit_of_each_layer)
#nn.learn(epoch = 30)
#nn.test()

rnn = network.recurrent_neural_network.RecurrentNeuralNetwork()
rnn.learn()
rnn.reinit()
rnn.learn()
