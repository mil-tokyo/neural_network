from abstract_neural_network import AbstractNeuralNetwork
import numpy as np
import layer

class NeuralNetwork(AbstractNeuralNetwork):

    def __init__(self, num_unit_of_each_layer, batch_size = 120):
        print "initializing network"

        self.load_config()
        #dataset_path = '/Users/tomoakisaito/Documents/dataset'
        dataset_path = '../dataset'
        self.load_images(dataset_path+'/mnist.pkl.gz')
        #self.load_mnist('/data/ishimochi0/dataset/mnist')
        
        #print self.train_labels
        self.num_unit_of_each_layer = num_unit_of_each_layer
        self.layers = []
        self.rate = 0.05
        self.activate_function = 'sigmoid'
        fc_num = len(num_unit_of_each_layer)

        '''
        initialize layers
        '''
        for i in range(fc_num-2):
            self.layers.append(layer.FullyConnectedLayer(num_unit_of_each_layer[i], num_unit_of_each_layer[i+1]))
            self.layers.append(layer.ActivateLayer(self.activate_function))

        self.layers.append(layer.FullyConnectedLayer(num_unit_of_each_layer[fc_num-2], num_unit_of_each_layer[fc_num-1]))
        self.layers.append(layer.ActivateLayer('softmax'))
        self.layers.append(layer.OutputLayer())

        self.layers_num = len(self.layers)

    def load_config(self):
#        super(ConvolutionalNeuralNetwork, self).test()
        print('load config')

    def learn(self, epoch=1):
        print "learn start"
        total_error = 0
        print_interval = 5000
        print "total epoch : "+str(epoch)
        for k in range(epoch):
            print "epoch : "+str(k+1)
            for i in xrange(len(self.train_labels)):
                if i % print_interval == print_interval - 1: 
                    print str(i+1)+" images learned   ,  ",

                # go forward
                self.layers[0].node = [self.train_images[i]]
                for j in xrange(self.layers_num-1):
                    self.layers[j+1].node = self.layers[j].forward()

                # go back
                err = self.layers[self.layers_num - 1].back(self.train_labels[i])
                total_error += err
                for j in range(1,self.layers_num):
                    self.layers[self.layers_num-j-1].back(next_node=self.layers[self.layers_num-j].node, next_derr=self.layers[self.layers_num-j].derr)

                if i % print_interval == print_interval - 1:
                    print "error: "+str(total_error/float(print_interval))
                    total_error = 0

                # update
                for j in range(0,self.layers_num-1):
                    self.layers[j].update(self.rate)
                #exit()
            self.test()

    def test(self):
#        super(ConvolutionalNeuralNetwork, self).test()
        print "test start"
        score = 0.

        for i in xrange(len(self.test_labels)):
#            if i % 100 == 0:
#                print str(i)+" images tested"
            # go forward
            self.layers[0].node = [self.test_images[i]]
            for j in xrange(self.layers_num-1):
                self.layers[j+1].node = self.layers[j].forward()

            out = self.layers[self.layers_num-1].node[0]
#            if i % 100 == 0:
#                print out
#                print self.test_labels[i]
            if np.argmax(out) == np.argmax(self.test_labels[i]):
                score += 1.

        score /= len(self.test_labels)
        score *= 100
        print "accuracy_rate = "+str(score)+"%"


    def save_network(self,file_path=None):
        print('save network')
        

    def load_network(self,file_path=None):
        print('load network')




