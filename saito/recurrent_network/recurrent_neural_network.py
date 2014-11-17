import numpy as np
import layer


class RecurrentNeuralNetwork():
    def __init__(self):
        print "initializing network"
        self.layers = []
        self.dic_size = 200000
        self.hidden_node_size = 100
        self.max_trancate = 100
        self.alpha = 0.05
        self.beta = 0.01

        self.layers.append(layer.InputLayer(self.dic_size, self.hidden_node_size))
        self.layers.append(layer.HiddenLayer(self.dic_size, self.hidden_node_size, 'sigmoid'))
        self.layers.append(layer.OutputLayer())

    def learn(self):
        sentence = np.random.rand(1000,self.dic_size)
        sentence = np.exp(sentence) / sum(np.exp(sentence))
        layers_num = len(self.layers)
        for word_index in xrange(sentence.shape[0]-1):
            print str(word_index) + " th word is being learned"
            '''
            forward
            '''
            self.layers[0].node = sentence[word_index]
            for i in xrange(layers_num-1):
                self.layers[i+1].node = self.layers[i].forward(self.max_trancate)
            '''
            backward
            '''
            self.layers[layers_num - 1].back(next_derr = sentence[word_index+1])
            for i in xrange(layers_num-1):
                self.layers[layers_num-2-i].back(next_derr = self.layers[layers_num-1-i].derr, max_trancate = self.max_trancate)

            '''
            update
            '''
            for i in xrange(layers_num-1):
                self.layers[i].update(self.alpha, self.beta)
        




    
