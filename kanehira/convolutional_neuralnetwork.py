from layers.convolutional_layer import ConvolutionalLayer
from layers.maxpooling_layer import MaxPooling
from layers.activation_layer import Activation
from layers.fully_connected_layer import FCLayer
import read_setting
import numpy as np
   
class NeuralNetwork:
    def __init__(self, network_setting):
        ## read network setting ##
        self.eta0 = network_setting["learning_rate"]
        self.iteration = network_setting["iteration"]
        self.batch_size = network_setting["batch_size"]
        layers_setting = network_setting["layers_setting"]

        self.layers_list = []
        for setting in layers_setting:
            layer_type = setting["layer_type"]
            print "initialize {}...".format(layer_type)
            setting_detail = setting["setting_detail"]
            layer = eval(layer_type+"(setting_detail)")
            self.layers_list.append(layer)

    def train(self, x_train, labels):
        """ 
        input -> x_train: train samples (datanum, featuredim), 
                 labels:label (datanum,classnum),
                 eta:training coefficiend
        """
        datanum = x_train.shape[0]
        likelihood = 0
        for j in xrange(self.iteration):
            self.eta = self.eta0 / ( 1 + j / (self.iteration-1)) 
            for i in xrange(datanum):
                if i % 1000 == 0:
                    print "data: {}/{} error:{}".format(i, datanum, likelihood / 1000)
                    likelihood = 0
                x = x_train[i, :]
                t = labels[i, :]
                """ForwardPropagetion"""
                output = self.forward_propagate(x)
                likelihood -= np.sum( t * np.log(output) )
                """BackPropagetion"""
                self.back_propagate(t, output)
                """Update parameters"""
                if i % self.batch_size == 0:
                    self.update()

    def predict(self, x_test, classnum):
        datanum = x_test.shape[0]
        result = np.zeros((datanum, classnum))
        for i in xrange(datanum):
            x = x_test[i, :]
            output = self.forward_propagate(x)
            result[i, np.argmax(output)] = 1
        return result
    
    def forward_propagate(self, x):
        """calculate forward process"""
        input = x
        for i,l in enumerate(self.layers_list):
            output = l.forward_calculate(input)
            input = output
        return output
    
    def back_propagate(self, t, output):
        """calculate back propagation"""
        prev_delta = output-t
        for i, l in enumerate(self.layers_list[::-1]):
            delta = l.back_calculate(prev_delta)
            prev_delta = delta
            
    def update(self):
        """ update all layers """
        for i, l in enumerate(self.layers_list):
            l.update(self.eta, self.batch_size)
           
    def __str__(self):
        """ show network information """
        return "\n".join(map(str, self.layers_list))
