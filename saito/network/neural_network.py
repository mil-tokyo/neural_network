from abstract_neural_network import AbstractNeuralNetwork
import numpy as np
import layer

class NeuralNetwork(AbstractNeuralNetwork):

    def __init__(self, num_unit_of_each_layer):
        print "initializing network"
#        self.load_config()
        # make layers
        self.load_config()
        dataset_path = '/Users/tomoakisaito/Documents/dataset'
        self.load_images(dataset_path+'/mnist.pkl.gz')
        print self.train_images.shape
        print self.train_labels
        self.num_unit_of_each_layer = num_unit_of_each_layer
        self.layers = []
        self.loss = layer.loss.Loss('square')
        self.rate = 0.05
        self.activate_function = 'sigmoid'

        for i in range(0,len(num_unit_of_each_layer)-1):
            self.layers.append(layer.fully_connected_layer.FullyConnectedLayer(num_unit_of_each_layer[i], self.activate_function))

        self.layers.append(layer.fully_connected_layer.FullyConnectedLayer(num_unit_of_each_layer[i], self.activate_function, is_output=True))

        for i in range(1,len(self.layers)):
            previous_layer_node = self.layers[i-1].node_num
            self.layers[i].init_weight(previous_layer_node)

    def load_config(self):
#        super(ConvolutionalNeuralNetwork, self).test()
        print('load config')

    def learn(self):
        print "learn start"
        total_error = 0
        layers_num = len(self.num_unit_of_each_layer)
        for i in range(0,len(self.train_labels)):
            if i % 100 == 0:                
                print str(i)+" images learned"
            # go forward
            input_node = self.train_images[i]            
            self.layers[0].forward(input_node)
            for j in range(1,layers_num):
                self.layers[j].forward(self.layers[j-1].node)

            # go back

            out = self.layers[layers_num-1].node
            label = self.train_labels[i]
            self.loss.loss(out,label)
            err = self.loss.error
            total_error += err
            if i%100 == 0:
                print "error: "+str(total_error/100.0)
                total_error = 0
            self.layers[layers_num-1].grad = self.loss.grad
            
            for j in range(1,layers_num):
                self.layers[layers_num-j-1].grad = self.layers[layers_num-j].back()
                self.layers[layers_num-j].dweight = np.array(np.mat(self.layers[layers_num-j].derr).T*np.mat(self.layers[layers_num-j-1].node))

            # update
            for j in range(1,layers_num):
                self.layers[j].weight = self.layers[j].weight - self.rate * self.layers[j].dweight
                self.layers[j].bias = self.layers[j].bias - self.rate * self.layers[j].dbias

    def test(self):
#        super(ConvolutionalNeuralNetwork, self).test()
        print "test start"
        score = 0.

        layers_num = len(self.num_unit_of_each_layer)
        for i in range(0,len(self.test_labels)):
            if i % 100 == 0:
                print str(i)+" images tested"
            # go forward
            input_node = self.test_images[i]
            self.layers[0].forward(input_node)
            for j in range(1,layers_num):
                self.layers[j].forward(self.layers[j-1].node)

            out = self.layers[layers_num-1].node
#            if i % 100 == 0:
#                print out
#                print self.test_labels[i]
            if out[self.test_labels[i]] == max(out):
                score += 1.

        score /= len(self.test_labels)
        score *= 100
        print "accuracy_rate = "+str(score)+"%"


    def save_network(self,file_path=None):
        print('save network')
        

    def load_network(self,file_path=None):
        print('load network')




