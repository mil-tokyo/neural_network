import numpy as np
from layers.fully_connected_layer import FCLayer
from layers.activation_layer import Activation
import sys
sys.path.append("../saito/")
from toolbox import load_language_model


class DummyLayer():
    def __init__(self):
        pass

    def forward_calculate(self, inp):
        return 0

    def back_calculate(self, delta):
        pass

    def get_params(self, param):
        return 0

class RecurrentNeurealNetwork():
    def __init__(self, network_setting):
        self.W = np.random.normal(0, 0.01, size=(50, 50))
        self.U = np.random.normal(0, 0.01, size=(50, 100))
        self.V = np.random.normal(0, 0.01, size=(10, 50))
        self.eta = 0.05#network_setting["learning_rate"]
        self.batch_size = 1#network_setting["batch_size"]
        self.network_depth = network_setting["network_depth"]

        #layers_setting = network_setting["layers_setting"]
        self.input_layers_list = []
        self.hidden_layers_list = []
        self.activation_layers_list = []
        for i in range(self.network_depth-1):
            self.input_layers_list.append(FCLayer({"input_num" : 100, "output_num" : 50}, init_W = self.U))
            self.hidden_layers_list.append(FCLayer({"input_num" : 50, "output_num" : 50}, init_W = self.W))
            self.activation_layers_list.append(Activation({"activation_type" : "sigmoid"}))
        self.hidden_layers_list.append(FCLayer({"input_num" : 50, "output_num" : 10}, init_W = self.V))
        self.activation_layers_list.append(Activation({"activation_type" : "softmax"}))
        self.input_layers_list.append(DummyLayer())
                                                                                     
    def train(self, x_train, labels):
        datanum = x_train.shape[0]
        iteration = 3
        for j in xrange(iteration):
            for i in xrange(datanum):
                if i % 1 == 0:
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
        inputs = np.vstack((x, np.zeros(x.shape[1])))
        old_hidden_output = np.zeros(50)
        for i in xrange(self.network_depth):
            input_layer_output = self.input_layers_list[i].forward_calculate(inputs[i, :])
            hidden_layer_output = self.hidden_layers_list[i].forward_calculate(old_hidden_output)
            old_hidden_output = self.activation_layers_list[i].forward_calculate(input_layer_output + hidden_layer_output)

        return old_hidden_output                                                        

    def back_propagate(self, t, output):
        """calculate back propagation"""
        print output
        prev_delta = output - t
        for i in range(self.network_depth)[::-1]:
            #print  prev_delta
            delta = self.activation_layers_list[i].back_calculate(prev_delta)
            delta = self.hidden_layers_list[i].back_calculate(delta)
            prev_delta = delta
                                                  
    def update(self):
        V_div = self.hidden_layers_list[-1].get_params("div")
        U_div = sum([layer.get_params("div") for layer in self.input_layers_list])
        W_div = sum([layer.get_params("div") for layer in self.hidden_layers_list[:-1]])
#        print V_div
        self.V -= self.eta * V_div + 0.1 * self.V
        self.U -= self.eta * U_div + 0.1 * self.U
        self.W -= self.eta * W_div + 0.1 * self.W


if __name__ == "__main__":
    rnn = RecurrentNeurealNetwork({"network_depth" : 3})
    train, test = load_language_model()
    print train
#    a = np.ones((1000, 2, 100))
#    t = np.zeros((1000, 10))
#    t[:, 0] = 1
#    rnn.train(a, t)


