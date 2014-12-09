import rcnn as rc_net
import numpy as np

#num_unit_of_each_layer = np.array([784, 128, 10])
#nn = network.neural_network.NeuralNetwork(num_unit_of_each_layer)
#nn.learn(epoch = 30)
#nn.test()

rcnn = rc_net.recurrent_convolutional_neural_network.DebugRecorrentConvolutionalNeuralNetwork()
rcnn.learn(epoch = 150000)
