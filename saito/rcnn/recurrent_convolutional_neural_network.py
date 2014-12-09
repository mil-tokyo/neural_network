from abstract_neural_network import AbstractNeuralNetwork
import numpy as np
import layer
import random
import time

class RecurrentConvolutionalNeuralNetwork(AbstractNeuralNetwork):
    def __init__(self):
        print "initializing network"

        self.load_config()
        self.load_ibc()
        self.layers = []
        self.activate_function = 'sigmoid'
        self.batch_size = 1
        self.select_frame = 6
        self.rates = [0.01, 0.01, 0.01]
        # rnn weight & cnn weight & nn weight

        #self.layers.append(layer.ActivateLayer(self.activate_function))
        self.layers.append(layer.RecurrentLayer(activate_function = self.activate_function, input_kernel = 256, kernel_shape = (3,3)))
        self.layers.append(layer.ConvolutionalLayer(num_input = 256,num_output = 128,kernel_shape = (4,4,3),stride = 1))
        self.layers.append(layer.PoolingLayer(kernel_shape = (2,2,2), stride_shape = (2,2,2), pool_type = 'MAX'))
        self.layers.append(layer.ActivateLayer(self.activate_function))
        self.layers.append(layer.ConvolutionalLayer(num_input = 128,num_output = 64,kernel_shape = (3,3,3),stride = 1))
        self.layers.append(layer.ActivateLayer(self.activate_function))
        self.layers.append(layer.SpacialPoolingLayer(select_frame = self.select_frame))
        self.layers.append(layer.FullyConnectedLayer(3*3*6*64,1024))
        self.layers.append(layer.ActivateLayer(self.activate_function))
        self.layers.append(layer.FullyConnectedLayer(1024,324))
        self.layers.append(layer.ActivateLayer(self.activate_function))
        self.layers.append(layer.FullyConnectedLayer(324,127))
        self.layers.append(layer.ActivateLayer('softmax'))
        self.layers.append(layer.OutputLayer())

        #self.rates[0] = self.rates[0] / self.select_frame

        self.layers_num = len(self.layers)

    def load_config(self):
#        super(ConvolutionalNeuralNetwork, self).test()
        print('load config')

    def learn(self, epoch=1):
        forward_times = np.zeros(self.layers_num)
        backward_times = np.zeros(self.layers_num)
        print "learn start"
        total_error = 0
        print_interval = 1
        test_interval = 5000
        print "total epoch : "+str(epoch)
        seed = [i for i in xrange(len(self.train_label))]
        for k in range(epoch):
            self.train_data = []
            epoch_seed = random.sample(seed, self.batch_size)
            for i in xrange(len(epoch_seed)):
                self.train_data.append(np.load(self.train_path[epoch_seed[i]]))
                #self.train_data.append(np.load('/data/ishimochi2/saito/IBC/decaf/conv5/80729.npy'))
                self.train_data[i] = self.train_data[i].reshape(self.train_data[i].shape[0], self.train_data[i].shape[2], self.train_data[i].shape[3], self.train_data[i].shape[4])
                self.train_data[i] = self.train_data[i].astype(np.float64)

                print self.train_path[epoch_seed[i]]

            for i in xrange(len(self.train_data)):
                if i % print_interval == print_interval - 1: 
                    print str(i+1)+" images learned   ,  ",

                # go forward
                self.layers[0].node = self.train_data[i]
                for j in xrange(self.layers_num-1):
                    start = time.time()
                    self.layers[j+1].node = self.layers[j].forward()
                    end = time.time()
                    forward_times[j]+=end - start

                # go back
                err = self.layers[self.layers_num - 1].back(self.train_label[epoch_seed[i]])
                total_error += err
                for j in range(1,self.layers_num):
                    start = time.time()
                    self.layers[self.layers_num-j-1].back(next_node=self.layers[self.layers_num-j].node, next_derr=self.layers[self.layers_num-j].derr)
                    end = time.time()
                    backward_times[self.layers_num-j-1]+=end - start
                    

            # update
            for j in range(0,self.layers_num-1):
                self.layers[j].update(self.rates, self.batch_size)
            #exit()
            if k % print_interval == print_interval - 1:
                total_error /= print_interval * self.batch_size
                print "epoch : "+str(k+1),
                print "  error : "+str(total_error),
                print "  frame_num : "+str(self.layers[0].node.shape[0])
                # print 'forward_time : '
                # for i in xrange(forward_times.shape[0]):
                #     print 'layer'+str(i)+': ',
                #     print forward_times[i]
                # print 'backward_time : '
                # for i in xrange(backward_times.shape[0]):
                #     print 'layer'+str(i)+': ',
                #     print backward_times[i]
                forward_times.fill(0)
                backward_times.fill(0)
                total_error = 0
            if k % test_interval == test_interval - 1:
                self.test()
                

    def test(self):
#        super(ConvolutionalNeuralNetwork, self).test()
        print "test start"
        score = 0.

        for i in xrange(len(self.test_label)):
#            if i % 100 == 0:
#                print str(i)+" images tested"
            # go forward
            self.layers[0].node = self.test_data[i]
            for j in xrange(self.layers_num-1):
                self.layers[j+1].node = self.layers[j].forward()

            out = self.layers[self.layers_num-1].node
#            if i % 100 == 0:
#                print out
#                print self.test_label[i]
            if np.argmax(out) == np.argmax(self.test_label[i]):
                score += 1.

        score /= len(self.test_label)
        score *= 100
        print "accuracy_rate = "+str(score)+"%"


    def save_network(self,file_path=None):
        print('save network')
        

    def load_network(self,file_path=None):
        print('load network')


class DebugRecorrentConvolutionalNeuralNetwork(AbstractNeuralNetwork):
    def __init__(self):
        print "initializing network"

        self.load_config()
        self.load_ibc()
        self.layers = []
        self.activate_function = 'sigmoid'
        self.batch_size = 1
        self.select_frame = 16
        self.rates = [0.04/self.select_frame, 0.05, 0.05]
        # rnn weight & cnn weight & nn weight

        self.layers.append(layer.FullyConnectedLayer(4096,1024))
        self.layers.append(layer.ActivateLayer(self.activate_function))
        self.layers.append(layer.FullyConnectedLayer(1024,256))
        self.layers.append(layer.ActivateLayer(self.activate_function))
        self.layers.append(layer.FullyConnectedLayer(256,127))
        self.layers.append(layer.ActivateLayer('softmax'))
        self.layers.append(layer.OutputLayer())

        self.layers_num = len(self.layers)

    def load_config(self):
#        super(ConvolutionalNeuralNetwork, self).test()
        print('load config')

    def learn(self, epoch=1):
        print "learn start"
        total_error = 0
        print_interval = 100
        test_interval = 8014
        print "total epoch : "+str(epoch)
        seed = [i for i in xrange(len(self.train_label))]
        for k in range(epoch):
            epoch_seed = random.sample(seed, self.batch_size)
            for i in xrange(len(epoch_seed)):
                if i % print_interval == print_interval - 1: 
                    print str(i+1)+" images learned   ,  ",

                # go forward
                self.layers[0].node = sum(self.train_data[epoch_seed[i]].T).reshape(4096)
                for j in xrange(self.layers_num-1):
                    self.layers[j+1].node = self.layers[j].forward()

                # go back
                err = self.layers[self.layers_num - 1].back(self.train_label[epoch_seed[i]])
                total_error += err
                for j in range(1,self.layers_num):
                    self.layers[self.layers_num-j-1].back(next_node=self.layers[self.layers_num-j].node, next_derr=self.layers[self.layers_num-j].derr)

            # update
            for j in range(0,self.layers_num-1):
                self.layers[j].update(self.rates, self.batch_size)
            #exit()
            if k % print_interval == print_interval - 1:
                total_error /= print_interval * self.batch_size
                print "epoch : "+str(k+1),
                print "  error : "+str(total_error)
                total_error = 0
            if k % test_interval == test_interval - 1:
                self.test()
                

    def test(self):
#        super(ConvolutionalNeuralNetwork, self).test()
        print "test start"
        score = 0.

        for i in xrange(len(self.test_label)):
#            if i % 100 == 0:
#                print str(i)+" images tested"
            # go forward
            self.layers[0].node = sum(self.test_data[i].T).reshape(4096)
            for j in xrange(self.layers_num-1):
                self.layers[j+1].node = self.layers[j].forward()

            out = self.layers[self.layers_num-1].node
#            if i % 100 == 0:
#                print out
#                print self.test_label[i]
            if np.argmax(out) == np.argmax(self.test_label[i]):
                score += 1.

        score /= len(self.test_label)
        score *= 100
        print "accuracy_rate = "+str(score)+"%"

    def save_network(self,file_path=None):
        print('save network')        

    def load_network(self,file_path=None):
        print('load network')


'''
class DebugRecorrentConvolutionalNeuralNetwork(AbstractNeuralNetwork):
    def __init__(self):
        print "initializing network"

        self.load_config()
        self.load_mnist('/data/ishimochi0/dataset/mnist')
        self.layers = []
        self.activate_function = 'sigmoid'
        self.batch_size = 1
        self.select_frame = 16
        self.rates = [0.04/self.select_frame, 0.05, 0.05]
        # rnn weight & cnn weight & nn weight

        self.layers.append(layer.ConvolutionalLayer(num_input = 1,num_output = 10,kernel_shape = (5,5),stride = 1))
        self.layers.append(layer.PoolingLayer(kernel_shape = (2,2), stride_shape = (2,2), pool_type = 'MAX'))
        self.layers.append(layer.ActivateLayer(self.activate_function))
        self.layers.append(layer.ConvolutionalLayer(num_input = 10,num_output = 12,kernel_shape = (3,3),stride = 1))
        self.layers.append(layer.PoolingLayer(kernel_shape = (2,2), stride_shape = (2,2), pool_type = 'MAX'))
        self.layers.append(layer.ActivateLayer(self.activate_function))
        self.layers.append(layer.FullyConnectedLayer(300,128))
        self.layers.append(layer.ActivateLayer(self.activate_function))
        self.layers.append(layer.FullyConnectedLayer(128,10))
        self.layers.append(layer.ActivateLayer('softmax'))
        self.layers.append(layer.OutputLayer())

        self.layers_num = len(self.layers)

    def load_config(self):
#        super(ConvolutionalNeuralNetwork, self).test()
        print('load config')

    def learn(self, epoch=1):
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
                    self.layers[j+1].node = self.layers[j].forward()

                # go back
                err = self.layers[self.layers_num - 1].back(self.train_labels[epoch_seed[i]])
                total_error += err
                for j in range(1,self.layers_num):
                    self.layers[self.layers_num-j-1].back(next_node=self.layers[self.layers_num-j].node, next_derr=self.layers[self.layers_num-j].derr)

            # update
            for j in range(0,self.layers_num-1):
                self.layers[j].update(self.rates, self.batch_size)
            #exit()
            if k % print_interval == print_interval - 1:
                total_error /= print_interval * self.batch_size
                print "epoch : "+str(k+1),
                print "  error : "+str(total_error)
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
#                print self.test_label[i]
            if np.argmax(out) == np.argmax(self.test_labels[i]):
                score += 1.

        score /= len(self.test_labels)
        score *= 100
        print "accuracy_rate = "+str(score)+"%"


    def save_network(self,file_path=None):
        print('save network')
        

    def load_network(self,file_path=None):
        print('load network')



'''
