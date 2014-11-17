import numpy as np
from layers.fully_connected_layer import FCLayer
from layers.activation_layer import Activation

class RcnnLayer():
    def __init__(self, U, W):
        self.input_layer = FCLayer(U)
        self.hidden_layer = FCLayer(W)
        
    def forward_calculate(self, inp, hidden):
        pass

class RecurrentNeurealNetwork():
    def __init__(self):
        self.W = np.random.normal(0, 0.05, size=(50, 50))
        self.U = np.random.normal(0, 0.05, size=(100, 50))
        self.V = np.random.normal(0, 0.05, size=(50, 10))
        #self.eta = network_setting["learning_rate"]
        #self.batch_size = network_setting["batch_size"]
        self.network_depth = network_setting["network_depth"]

        #layers_setting = network_setting["layers_setting"]
        
        self.input_layers_list = []
        self.hidden_layers_list = []
        for i in range(self.network_depth-1):
            self.input_layers_list.append(FCLayer({"input_num" : 100, "output_num" : 50}, init_W = self.U))
            self.hidden_layers_list.append(FCLayer({"input_num" : 50, "output_num" : 50}, init_W = self.W))
            self.activation_layers_list.append(Activation({"activation_type" : "sigmoid"}))
        self.hidden_layers_list.append(FCLayer({"input_num" : 50, "output_num" : 10}, init_W = self.V))
        self.activation_layers_list.append(Activation({"activation_type" : "softmax"}))
                                                                                     
    def train(self, x_train, labels):
    """
    input -> x_train: train samples (datanum, featuredim),
    labels:label (datanum,classnum),
    eta:training coefficiend
    """
    datanum = x_train.shape[0]
    iteration = 3
    for j in xrange(iteration):
        for i in xrange(datanum):
            if i % 1000 == 0:
                print "data: %d/%d"%(i, datanum)
            x = x_train[i, :]
            t = labels[i, :]
            """ForwardPropagetion"""
            output = self.forward_propagate(x)
            """BackPropagetion"""
            self.back_propagate(t, output)
            """Update parameters"""
            if i % self.batch_size == 0:
                self.update()

                
    def predict(self):
        pass
        
    def forward_propagate(self, x):
        """calculate forward process"""
        inputs = x
        old_hidden_output = np.zeros(50)
        for i in xrange(self.network_depth-1):
            input_layer_output = self.input_layers_list[i].forward_calculate(inputs[i, :])
            hidden_layer_output = self.hidden_layers_list[i].forward_calculate(old_hidden_output)
            old_hidden_output = self.activation_layers_list[i].forward_calculate(input_layer_output + hidden_layer_output)

        output = self.hidden_layers_list[-1].forward_propagate(old_hidden_output)
        output = self.activation_layers_list[-1].forward_propagate(output)
        return output                                                        

    def back_propagate(self, t, output):
        """calculate back propagation"""
        prev_delta = output-t
        for i in range(0, self.network_depth, -1):
            delta = self.activation_layers_list[i].back_calculate(delta)
            delta = self.hidden_layers_list[i].back_calculate(delta)
            prev_delta = delta
                                                  
    def update(self):
        pass


if __name__ == "__main__":
    rnn = RecurrentNeurealNetwork({"network_depth" : 3})
    a = np.arange(200).reshape(2,100)
    print rnn.forward_propagate(a)
    
