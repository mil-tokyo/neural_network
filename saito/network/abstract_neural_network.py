from abc import ABCMeta, abstractmethod
import cPickle, gzip, numpy, sys, math

class AbstractNeuralNetwork(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def load_config(self):
        raise NotImplementedError()
        
    @abstractmethod
    def learn(self):
        raise NotImplementedError()

    @abstractmethod
    def test(self):
        raise NotImplementedError()

    def load_images(self, data_path):
        f=gzip.open(data_path,'rb')
        train_set, valid_set, test_set=cPickle.load(f)
        train_images, train_labels=train_set
        test_images, test_labels=test_set
#        dsize = int(math.sqrt(len(train_images[0])))
#        train_images = train_images.reshape(len(train_images),dsize,dsize)
#        dsize = int(math.sqrt(len(test_images[0])))
#        test_images = test_images.reshape(len(test_images),dsize,dsize)
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels


        # for i in range(data_index,data_index+1):
        #     for y in range(0,28):
        #         for x in range(0,28):
        #             if train_set_x[i][y*28+x]<0.5:
        #                 sys.stdout.write(" ")
        #             elif train_set_x[i][y*28+x]<0.8:
        #                 sys.stdout.write("+")
        #             else:
        #                 sys.stdout.write("*")
        #             sys.stdout.write("\n")
        # print "correct =",train_set_y[i]
        # print "--------------------------------"
