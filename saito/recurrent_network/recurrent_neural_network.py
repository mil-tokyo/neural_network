import numpy as np
import layer
import os


class RecurrentNeuralNetwork():
    def __init__(self):
        print "initializing network"
        self.layers = []
        self.max_trancate = 100
        self.alpha = 0.05
        self.beta = 0.01
        self.load_language_model('../dataset/rnnlm-data')
        self.dic_size = self.train_data.shape[1]
        self.hidden_node_size = 100

        self.layers.append(layer.InputLayer(self.dic_size, self.hidden_node_size))
        self.layers.append(layer.HiddenLayer(self.dic_size, self.hidden_node_size, 'sigmoid'))
        self.layers.append(layer.OutputLayer())

    def learn(self):
        print_interval = 500
        layers_num = len(self.layers)
        error = 0
        for word_index in xrange(self.train_data.shape[0]-1):
            if word_index % print_interval == print_interval - 1:
                print str(word_index + 1) + " th word is being learned"
            '''
            forward
            '''
            self.layers[0].node = self.train_data[word_index]
            for i in xrange(layers_num-1):
                self.layers[i+1].node = self.layers[i].forward(self.max_trancate)
            '''
            backward
            '''
            self.layers[layers_num - 1].back(next_derr = self.train_data[word_index+1])
            for i in xrange(layers_num-1):
                self.layers[layers_num-2-i].back(next_derr = self.layers[layers_num-1-i].derr, max_trancate = self.max_trancate)
            '''
            update
            '''
            for i in xrange(layers_num-1):
                self.layers[i].update(self.alpha, self.beta)
            error += self.layers[layers_num-1].error()
            if word_index % print_interval == print_interval - 1:
                print "error : " + str(error)
                error = 0
        
    def reinit(self):
        for i in range(len(self.layers)):
            self.layers[i].reinit()

    def load_language_model(self, datapath = '../dataset/rnnlm-data/'):
        train_file_path = os.path.join(datapath, 'ptb.train.txt')
        test_file_path = os.path.join(datapath, 'ptb.test.txt')
        train_txt = []
        test_txt = []
        print 'loading language dataset'
        print '   loading train data'
        for line in open(train_file_path,'r'):
            line_list = line.split(' ')
            for i in xrange(1,len(line_list)-1):
                train_txt.append(line_list[i])

        print '   loading test data'
        for line in open(test_file_path,'r'):
            line_list = line.split(' ')
            for i in xrange(1,len(line_list)-1):
                test_txt.append(line_list[i])

        print '   making dictionary'
        language_dictionary = []
        for i in xrange(len(train_txt)):
            if train_txt[i] not in language_dictionary:
                language_dictionary.append(train_txt[i])

        self.train_data = np.zeros((len(train_txt),len(language_dictionary)))
        self.test_data = np.zeros((len(test_txt),len(language_dictionary)))

        print '   making train&test data from dictionary'
        for i in xrange(len(train_txt)):
            self.train_data[i,language_dictionary.index(train_txt[i])] = 1
        for i in xrange(len(test_txt)):
            self.test_data[i,language_dictionary.index(test_txt[i])] = 1

        '''
        train_data : train data count * dictionary size
        test_data : test data count * dictionary size
        '''

    
