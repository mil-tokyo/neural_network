import numpy as np
import layer
import os


class RecurrentNeuralNetwork():
    def __init__(self):
        print "initializing network"
        self.layers = []
        self.max_trancate = 4
        self.alpha = 0.01
        self.beta = 0
        self.load_language_model('../dataset/rnnlm-data')
        self.dic_size = len(self.language_dictionary)
        self.hidden_node_size = 100

        self.layers.append(layer.InputLayer(self.dic_size, self.hidden_node_size))
        self.layers.append(layer.HiddenLayer(self.dic_size, self.hidden_node_size, 'sigmoid'))
        self.layers.append(layer.OutputLayer())

    def learn(self):
        print 'learn start'
        print_interval = 500
        layers_num = len(self.layers)
        error = 0
        word = None
        next_word = None
        for word_index in xrange(len(self.train_index) - 1):
            if word_index % print_interval == print_interval - 1:
                print str(word_index + 1) + " th word is being learned"

            if word is None:
                word = np.zeros(self.dic_size)
                word[self.train_index[word_index]] = 1
            else:
                word = next_word

            next_word = np.zeros(len(self.language_dictionary))
            next_word[self.train_index[word_index+1]] = 1
            '''
            forward
            '''
            self.layers[0].node = word
            for i in xrange(layers_num-1):
                self.layers[i+1].node = self.layers[i].forward(self.max_trancate)
            '''
            backward
            '''
            self.layers[layers_num - 1].back(next_derr = next_word)
            for i in xrange(layers_num-1):
                self.layers[layers_num-2-i].back(next_derr = self.layers[layers_num-1-i].derr, max_trancate = self.max_trancate)
            '''
            update
            '''
            for i in xrange(layers_num-1):
                self.layers[i].update(self.alpha, self.beta)
            error += self.layers[layers_num-1].error
            if word_index % print_interval == print_interval - 1:
                print "error : " + str(error / print_interval)
                error = 0

    def test(self):
        print 'test start'
        print_interval = len(self.test_index)
        layers_num = len(self.layers)
        error = 0
        word = None
        next_word = None
        for word_index in xrange(len(self.test_index) - 1):
            if word_index % print_interval == print_interval - 1:
                print str(word_index + 1) + " th word is being learned"

            if word is None:
                word = np.zeros(self.dic_size)
                word[self.test_index[word_index]] = 1
            else:
                word = next_word

            next_word = np.zeros(len(self.language_dictionary))
            next_word[self.test_index[word_index+1]] = 1
            '''
            forward
            '''
            self.layers[0].node = word
            for i in xrange(layers_num-1):
                self.layers[i+1].node = self.layers[i].forward(self.max_trancate)
            '''
            backward
            '''
            self.layers[layers_num - 1].back(next_derr = next_word)

            error += self.layers[layers_num-1].error
            if word_index % print_interval == print_interval - 1:
                print "error : " + str(error / print_interval)
                error = 0
        

        
    def reinit(self):
        for i in range(len(self.layers)):
            self.layers[i].reinit()

    def load_language_model(self, datapath = '../dataset/rnnlm-data/'):
        train_file_path = os.path.join(datapath, 'ptb.train.txt')
        test_file_path = os.path.join(datapath, 'ptb.test.txt')
        train_txt = []
        test_txt = []
        self.train_index = []
        self.test_index = []

        print 'loading language dataset'
        print '   loading train data'
        for line in open(train_file_path,'r'):
            line_list = line.split(' ')
            for i in xrange(1,len(line_list)):
                train_txt.append(line_list[i])

        print '   loading test data'
        for line in open(test_file_path,'r'):
            line_list = line.split(' ')
            for i in xrange(1,len(line_list)-1):
                test_txt.append(line_list[i])

        print '   making dictionary'
        self.language_dictionary = []
        for i in xrange(len(train_txt)):
            if train_txt[i] not in self.language_dictionary:
                self.language_dictionary.append(train_txt[i])
        print '      dic size : '+str(len(self.language_dictionary))

        print '   making train&test data from dictionary'
        for i in xrange(len(train_txt)):
            self.train_index.append(self.language_dictionary.index(train_txt[i]))
        for i in xrange(len(test_txt)):
            self.test_index.append(self.language_dictionary.index(test_txt[i]))
        '''
        train_data : train data count * dictionary size
        test_data : test data count * dictionary size
        '''

    
