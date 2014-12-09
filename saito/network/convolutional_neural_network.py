from abstract_neural_network import AbstractNeuralNetwork
import numpy as np
import layer
import random
import time

class ConvolutionalNeuralNetwork(AbstractNeuralNetwork):

    def __init__(self):
        print "initializing network"

        mode = 'mnist'
        self.load_config()
        #dataset_path = '/Users/tomoakisaito/Documents/dataset'
        dataset_path = '../dataset'
        #self.load_images(dataset_path+'/mnist.pkl.gz')
        #print self.train_labels
        self.layers = []
        self.rate = 0.05
        self.activate_function = 'sigmoid'
        self.batch_size = 1
        print mode
        print 'learning rate : ',
        print self.rate
        '''
        initialize layers
        '''
        if mode == 'mnist':
            self.load_mnist('/data/ishimochi0/dataset/mnist')
            self.layers.append(layer.ConvolutionalLayer(num_input = 1,num_output = 16,kernel_shape = (5,5),stride = 1))
            self.layers.append(layer.PoolingLayer(kernel_shape = (2,2), stride_shape = (2,2), pool_type = 'MAX'))
            self.layers.append(layer.ActivateLayer(self.activate_function))
            self.layers.append(layer.ConvolutionalLayer(num_input = 16,num_output = 32,kernel_shape = (3,3),stride = 1))
            self.layers.append(layer.PoolingLayer(kernel_shape = (2,2), stride_shape = (2,2), pool_type = 'MAX'))
            self.layers.append(layer.ActivateLayer(self.activate_function))
            self.layers.append(layer.ConvolutionalLayer(num_input = 32,num_output = 64,kernel_shape = (2,2),stride = 1,mode='valid'))
            self.layers.append(layer.PoolingLayer(kernel_shape = (2,2), stride_shape = (2,2), pool_type = 'MAX'))
            self.layers.append(layer.ActivateLayer(self.activate_function))
            self.layers.append(layer.ConvolutionalLayer(num_input = 64,num_output = 128,kernel_shape = (2,2),stride = 1,mode='valid'))
            self.layers.append(layer.ActivateLayer(self.activate_function))
            self.layers.append(layer.FullyConnectedLayer(128,32))
            self.layers.append(layer.ActivateLayer(self.activate_function))
            self.layers.append(layer.FullyConnectedLayer(32,10))
            self.layers.append(layer.ActivateLayer('softmax'))
            self.layers.append(layer.OutputLayer())

        if mode == 'cifar':
            self.load_cifar('/data/ishimochi2/saito/cifar-10-batches-py')
            self.layers.append(layer.ConvolutionalLayer(num_input = 3,num_output = 12,kernel_shape = (5,5),stride = 1))
            self.layers.append(layer.PoolingLayer(kernel_shape = (2,2), stride_shape = (2,2), pool_type = 'MAX'))
            self.layers.append(layer.ActivateLayer(self.activate_function))
            self.layers.append(layer.ConvolutionalLayer(num_input = 12,num_output = 24,kernel_shape = (3,3),stride = 1))
            self.layers.append(layer.PoolingLayer(kernel_shape = (2,2), stride_shape = (2,2), pool_type = 'MAX'))
            self.layers.append(layer.ActivateLayer(self.activate_function))
            self.layers.append(layer.ConvolutionalLayer(num_input = 24,num_output = 48,kernel_shape = (3,3),stride = 1,mode='same'))
            self.layers.append(layer.PoolingLayer(kernel_shape = (2,2), stride_shape = (2,2), pool_type = 'MAX'))
            self.layers.append(layer.ActivateLayer(self.activate_function))
            self.layers.append(layer.ConvolutionalLayer(num_input = 48,num_output = 96,kernel_shape = (3,3),stride = 1,mode='valid'))
            self.layers.append(layer.ActivateLayer(self.activate_function))
            self.layers.append(layer.FullyConnectedLayer(96,32))
            self.layers.append(layer.ActivateLayer(self.activate_function))
            self.layers.append(layer.FullyConnectedLayer(32,10))
            self.layers.append(layer.ActivateLayer('softmax'))
            self.layers.append(layer.OutputLayer())

        self.layers_num = len(self.layers)

    def load_config(self):
#        super(ConvolutionalNeuralNetwork, self).test()
        print('load config')

    def learn(self, epoch=1):
        forward_times = np.zeros(self.layers_num)
        backward_times = np.zeros(self.layers_num)
        print "learn start"
        total_error = 0
        print_interval = 1000
        test_interval = 50000
        print "total epoch : "+str(epoch)
        seed = [i for i in xrange(len(self.train_labels))]
        for k in range(epoch):
            epoch_seed = random.sample(seed, self.batch_size)
            for i in xrange(len(epoch_seed)):
                if i % print_interval == print_interval - 1: 
                    print str(i+1)+" images learned   ,  ",
                
                # go forward
                self.layers[0].node = self.train_images[epoch_seed[i]]
                for j in xrange(self.layers_num-1):
                    start = time.time()
                    self.layers[j+1].node = self.layers[j].forward()
                    end = time.time()
                    forward_times[j]+=end - start

                # go back for last layer
                err = self.layers[self.layers_num - 1].back(self.train_labels[epoch_seed[i]])
                total_error += err

                # go back 
                for j in range(1,self.layers_num):
                    start = time.time()
                    self.layers[self.layers_num-j-1].back(next_node=self.layers[self.layers_num-j].node, next_derr=self.layers[self.layers_num-j].derr)
                    end = time.time()
                    backward_times[self.layers_num-j-1]+=end - start

            for j in range(0,self.layers_num-1):
                self.layers[j].update(self.rate, negative_count)
            #exit()

            if k % print_interval == print_interval - 1:
                total_error /= print_interval * self.batch_size
                print "epoch : "+str(k+1),
                print "  error : "+str(total_error)
                # print 'forward_time : '
                # for i in xrange(forward_times.shape[0]):
                #     print 'layer'+str(i)+': ',
                #     print forward_times[i]
                # print 'backward_time : '
                # for i in xrange(backward_times.shape[0]):
                #     print 'layer'+str(i)+': ',
                #     print backward_times[i]
                forward_times = np.zeros(forward_times.shape)
                backward_times = np.zeros(backward_times.shape)
                total_error = 0
                
            if k % test_interval == test_interval - 1:
                self.test()

    def test(self):
#        super(ConvolutionalNeuralNetwork, self).test()
        print "test start"
        score = 0.

        for i in xrange(len(self.test_labels)):
#            if i % 100 == 0:
#                print str(i)+" images tested"
            # go forward
            self.layers[0].node = self.test_images[i]
            for j in xrange(self.layers_num-1):
                self.layers[j+1].node = self.layers[j].forward()

            out = self.layers[self.layers_num-1].node
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




