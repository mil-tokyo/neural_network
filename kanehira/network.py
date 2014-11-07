from layers.convolutional_layer import ConvolutionalLayer
from layers.maxpooling_layer import MaxPooling
from layers.activation_layer import Activation
from layers.fully_connected_layer import FCLayer
import read_setting
import numpy as np
   
class NeuralNetwork:
    def __init__(self, network_setting):
        ## read network setting ##
        self.eta = 0.01
        self.layers_list = []
        for setting in network_setting:
            layer_type = setting["layer_type"]
            print "initialize {}...".format(layer_type)
            setting_detail = setting["setting_detail"]
            layer = eval(layer_type+"(setting_detail)")
            self.layers_list.append(layer)

    def train(self, x_train, labels):
        ## x_train:train samples (datanum,featuredim), labels:label (datanum,classnum), eta:training coefficiend ##
        datanum = x_train.shape[0]
        iteration = 1
        for j in xrange(iteration):
            for i in xrange(datanum):
                if i % 1000 == 0:
                    print "data: %d/%d"%(i, datanum)
                x = x_train[i, :]
                t = labels[i, :]
                """ForwardPropagetion"""
                output = self.ForwardPropagate(x)
                """BackPropagetion"""
                self.BackPropagate(t, output)
                """Update parameters"""
                self.Update()

    def predict(self, x_test, classnum):
        datanum = x_test.shape[0]
        result = np.zeros((datanum, classnum))
        for i in xrange(datanum):
            x = x_test[i, :]
            output = self.ForwardPropagate(x)
            result[i, np.argmax(output)] = 1
        return result
    
    def ForwardPropagate(self, x):
        """calculate forward process"""
        input = x
        for i,l in enumerate(self.layers_list):
            output = l.forward_calculate(input)
            #if i==0:
            #    print np.sum(output)
            input = output
        return output
    
    def BackPropagate(self, t, output):
        """calculate back propagation"""
        prev_delta = output-t
        for i, l in enumerate(self.layers_list[::-1]):
            delta = l.back_calculate(prev_delta)
            prev_delta = delta
            
    def Update(self):
        """ update all layers """
        for i, l in enumerate(self.layers_list):
            l.update(self.eta)
           
    def __str__(self):
        """ show network information """
        return "\n".join(map(str, self.layers_list))
