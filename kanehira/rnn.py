import numpy as np
from layers.fully_connected_layer import FCLayer
from layers.activation_layer import Activation
from layers.dummy_layer import DummyLayer
import sys
sys.path.append("../saito/")
from toolbox import load_language_model


class RecurrentNeurealNetwork():
    def __init__(self, network_setting):
        self.alpha = 0.01#network_setting["learning_rate"]
        self.beta = 0.001
        self.batch_size = 1#network_setting["batch_size"]
        self.network_depth = network_setting["network_depth"]
        self.hidden_num = network_setting["hidden_num"]
        self.words_num = network_setting["input_num"]

        self.W = np.random.normal(0, 0.01, size=(self.hidden_num, self.hidden_num))
        self.U = np.random.normal(0, 0.01, size=(self.hidden_num, self.words_num))
        self.V = np.random.normal(0, 0.01, size=(self.words_num, self.hidden_num))

        #layers_setting = network_setting["layers_setting"]
        self.input_layers_list = []
        self.hidden_layers_list = []
        self.activation_layers_list = []
        for i in range(self.network_depth-1):
            self.input_layers_list.append(FCLayer({"input_num" : self.words_num, "output_num" : self.hidden_num}, init_W = self.U))
            self.hidden_layers_list.append(FCLayer({"input_num" : self.hidden_num, "output_num" : self.hidden_num}, init_W = self.W))
            self.activation_layers_list.append(Activation({"activation_type" : "sigmoid"}))
        self.hidden_layers_list.append(FCLayer({"input_num" : self.hidden_num, "output_num" : self.words_num}, init_W = self.V))
        self.activation_layers_list.append(Activation({"activation_type" : "softmax"}))
        self.input_layers_list.append(DummyLayer())
                                                                                     
    def train(self, x_train):
        datanum = x_train.shape[0]
        iteration = 1
        likelihood = 0
        for j in xrange(iteration):
            for i in xrange(datanum - self.network_depth):
                if i % 1000 == 0:
                    print "data: %d/%d likelihood: %f"%(i, datanum, likelihood / 1000)
                    likelihood = 0
                x = x_train[i:i+2, :]
                t = x_train[i+2, :]

                """ForwardPropagetion"""
                output = self.forward_propagate(x)

                """claclulate negative log likelihood"""
                likelihood -= np.sum(t * np.log(output))

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
        old_hidden_output = np.zeros(self.hidden_num)
        for i in range(self.network_depth):
            input_layer_output = self.input_layers_list[i].forward_calculate(inputs[i, :])
            hidden_layer_output = self.hidden_layers_list[i].forward_calculate(old_hidden_output)
            old_hidden_output = self.activation_layers_list[i].forward_calculate(input_layer_output + hidden_layer_output)

        return old_hidden_output                                                        

    def back_propagate(self, t, output):
        """calculate back propagation"""
        prev_delta = output - t
        for i in range(self.network_depth)[::-1]:
            delta = self.activation_layers_list[i].back_calculate(prev_delta)
            delta = self.hidden_layers_list[i].back_calculate(delta)
            prev_delta = delta
                                                  
    def update(self):
        V_div = self.hidden_layers_list[-1].get_params("div")
        U_div = sum([layer.get_params("div") for layer in self.input_layers_list[:-1]])
        W_div = sum([layer.get_params("div") for layer in self.hidden_layers_list[:-1]])

        self.V = self.V - self.alpha * V_div# - self.beta * self.V
        self.U = self.U - self.alpha * U_div# - self.beta * self.U
        self.W = self.W - self.alpha * W_div# - self.beta * self.W

        [layer.set_params("div", 0) for layer in self.input_layers_list[:-1]]
        [layer.set_params("div", 0) for layer in self.hidden_layers_list]

        self.hidden_layers_list[-1].set_params("W", self.V)
        [layer.set_params("W", self.U) for layer in self.input_layers_list[:-1]]
        [layer.set_params("W", self.W) for layer in self.hidden_layers_list[:-1]]

if __name__ == "__main__":
    train, test = load_language_model()
        
    rnn = RecurrentNeurealNetwork({"input_num": train.shape[1], "hidden_num" : 100, "network_depth" : 3})
#    a = np.ones((1000, 2, 100))
#    t = np.zeros((1000, 10))
#    t[:, 0] = 1
    rnn.train(train)


