from abstract_neural_network import AbstractNeuralNetwork
import numpy as np
import layer

class NeuralNetwork(AbstractNeuralNetwork):

    def __init__(self, num_unit_of_each_layer, batch_size = 120):
        print "initializing network"
        #self.load_config()
        # make layers
        self.load_config()
        #dataset_path = '/Users/tomoakisaito/Documents/dataset'
        dataset_path = '../dataset'
        #self.load_images(dataset_path+'/mnist.pkl.gz')
        self.load_mnist('/data/ishimochi0/dataset/mnist')

        # print "label : "+str(self.test_labels[10])
        # for i in range(0,28):
        #     for j in range(0,28):
        #         print self.test_images[10,i*28+j],
        #         # if self.test_images[10,i*28+j]>0.5:
        #         #     print "*",
        #         # else:
        #         #     print "-",
        #     print ""
        # print self.train_images.shape
        
        print self.train_labels
        self.num_unit_of_each_layer = num_unit_of_each_layer
        self.layers = []
        self.loss = layer.loss.Loss('square')
        self.rate = 0.05
        self.activate_function = 'sigmoid'

        for i in range(0,len(num_unit_of_each_layer)):
            self.layers.append(layer.fully_connected_layer.FullyConnectedLayer(num_unit_of_each_layer[i], self.activate_function))
        self.layers[len(num_unit_of_each_layer)-2].is_softmax = True

        for i in range(0,len(self.layers)-1):
            next_layer_node = self.layers[i+1].node_num
            self.layers[i].init_weight(next_layer_node)

    def load_config(self):
#        super(ConvolutionalNeuralNetwork, self).test()
        print('load config')

    def learn(self, epoch=1):
        print "learn start"
        total_error = 0
        layers_num = len(self.num_unit_of_each_layer)
        print_interval = 5000
        print "total epoch : "+str(epoch)
        for k in range(0,epoch):
            print "epoch : "+str(k+1)
            for i in range(0,len(self.train_labels)):
                if i % print_interval == 0:                
                    print str(i)+" images learned   ,  ",

                # go forward
                self.layers[0].node = self.train_images[i]
                for j in range(0,layers_num-1):
                    self.layers[j+1].node = self.layers[j].forward()

                # go back
                out = self.layers[layers_num-1].node
                label = self.train_labels[i]
                self.loss.loss(out,label)
                err = self.loss.error
                total_error += err
                if i%print_interval == 0:
                    print "error: "+str(total_error/float(print_interval))
                    total_error = 0
                self.layers[layers_num-1].grad = self.loss.grad
                self.layers[layers_num-1].back()
            
                for j in range(1,layers_num):
                    self.layers[layers_num-j-1].back(self.layers[layers_num-j].derr)

                # update
                for j in range(0,layers_num-1):
                    self.layers[j].update(self.rate)
            self.test()

    def test(self):
#        super(ConvolutionalNeuralNetwork, self).test()
        print "test start"
        score = 0.

        layers_num = len(self.num_unit_of_each_layer)
        for i in range(0,len(self.test_labels)):
#            if i % 100 == 0:
#                print str(i)+" images tested"
            # go forward
            self.layers[0].node = self.test_images[i]
            for j in range(0,layers_num-1):
                self.layers[j+1].node = self.layers[j].forward()

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




