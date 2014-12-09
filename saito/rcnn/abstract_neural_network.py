from abc import ABCMeta, abstractmethod
import cPickle, gzip, sys, math, os, struct
from array import array
import numpy as np

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
        dsize = int(math.sqrt(len(train_images[0])))
        train_images = train_images.reshape(len(train_images),1,dsize,dsize)
        dsize = int(math.sqrt(len(test_images[0])))
        test_images = test_images.reshape(len(test_images),1,dsize,dsize)

        self.train_images = train_images
        self.train_labels = np.fromfunction(lambda i,j:j==train_labels[i],(train_labels.size,max(train_labels)+1),dtype=int)+0
        self.test_images = test_images
        self.test_labels = test_labels
        self.test_labels = np.fromfunction(lambda i,j:j==test_labels[i],(test_labels.size,max(test_labels)+1),dtype=int)+0

    def load_mnist(self, data_path):
        fname_train_img = os.path.join(data_path, 'train-images-idx3-ubyte')
        fname_train_lbl = os.path.join(data_path, 'train-labels-idx1-ubyte')
        fname_test_img = os.path.join(data_path, 't10k-images-idx3-ubyte')
        fname_test_lbl = os.path.join(data_path, 't10k-labels-idx1-ubyte')

        # load test image
        flbl = open(fname_test_lbl, 'rb')
        magic_nr, size = struct.unpack(">II", flbl.read(8))
        test_lbl = array("b", flbl.read())
        flbl.close()

        fimg = open(fname_test_img, 'rb')
        magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
        test_img = array("B", fimg.read())
        fimg.close()
        
        # convert test image
        digits = np.arange(10)
        ind = [ k for k in range(size) if test_lbl[k] in digits]
        N = len(ind)

        test_images = np.zeros((N, rows*cols),dtype=np.uint8)
        test_labels = np.zeros(N,dtype=np.uint8)
        for i in range(len(ind)):
            test_images[i] = np.array(test_img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ])
            test_labels[i] = test_lbl[ind[i]]

        # load training image
        flbl = open(fname_train_lbl, 'rb')
        magic_nr, size = struct.unpack(">II", flbl.read(8))
        train_lbl = array("b", flbl.read())
        flbl.close()

        fimg = open(fname_train_img, 'rb')
        magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
        train_img = array("B", fimg.read())
        fimg.close()

        # convert training image
        digits = np.arange(10)
        ind = [ k for k in range(size) if train_lbl[k] in digits]
        N = len(ind)

        train_images = np.zeros((N, rows*cols),dtype=np.uint8)
        train_labels = np.zeros(N,dtype=np.uint8)
        for i in range(len(ind)):
            train_images[i] = np.array(train_img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ])
            train_labels[i] = train_lbl[ind[i]]

        test_images = test_images / 255.0
        train_images = train_images / 255.0

        # resize images to 2D
        dsize = int(math.sqrt(len(train_images[0])))
        train_images = train_images.reshape(len(train_images),1,dsize,dsize)
        dsize = int(math.sqrt(len(test_images[0])))
        test_images = test_images.reshape(len(test_images),1,dsize,dsize)

        self.test_images = test_images
        self.test_labels = np.fromfunction(lambda i,j:j==test_labels[i],(test_labels.size,max(test_labels)+1),dtype=int)+0
        self.train_images = train_images
        self.train_labels = np.fromfunction(lambda i,j:j==train_labels[i],(train_labels.size,max(train_labels)+1),dtype=int)+0

    
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
    def load_cifar(self,data_path):
        image_list = []
        label_list = []
        for i in xrange(5):
            fo = open(os.path.join(data_path,'data_batch_'+str(i+1)),'rb')
            dict = cPickle.load(fo)
            fo.close()
            image_list.append(dict['data'])
            label_list.append(dict['labels'])
        train_images = np.r_[image_list[0],image_list[1],image_list[2],image_list[3],image_list[4]]
        train_labels = np.r_[label_list[0],label_list[1],label_list[2],label_list[3],label_list[4]]
        fo = open(os.path.join(data_path,'test_batch'), 'rb')
        dict= cPickle.load(fo)
        fo.close()
        test_images = dict['data']
        test_labels = dict['labels']

        test_images = test_images / 255.0
        train_images = train_images / 255.0

        # resize images to 2D
        dsize = 32
        train_images = train_images.reshape(len(train_images),3,dsize,dsize)
        test_images = test_images.reshape(len(test_images),3,dsize,dsize)
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)
        
        self.test_images = test_images
        self.test_labels = np.fromfunction(lambda i,j:j==test_labels[i],(len(test_labels),max(test_labels)+1),dtype=int)+0
        self.train_images = train_images
        self.train_labels = np.fromfunction(lambda i,j:j==train_labels[i],(len(train_labels),max(train_labels)+1),dtype=int)+0


    def load_ibc(self,data_path = '/data/ishimochi2/saito/IBC/decaf/conv5/'):
        print 'load ibc data'
        self.train_name = np.load('./rcnn/train_name.npy')
        self.test_name = np.load('./rcnn/test_name.npy')
        self.train_label = np.load('./rcnn/train_label.npy')
        self.test_label = np.load('./rcnn/test_label.npy')

        self.train_path = []
        self.test_path = []
        self.train_data = []
        self.test_data = []
        ac = lambda x : 1.0/(1+np.exp(-x))
        
        for i in xrange(self.train_name.shape[0]):
            self.train_path.append(os.path.join(data_path,self.train_name[i]+'.npy'))
            # feature_path = os.path.join(data_path,self.train_name[i]+'.npy')
            # feature = np.load(feature_path)
            # feature = feature.reshape((feature.shape[0],feature.shape[2])).T
            # # feature : 4096 * frame
            # self.train_data.append(ac(feature))

        for i in xrange(self.test_name.shape[0]):
            self.test_path.append(os.path.join(data_path, self.test_name[i]+'.npy'))
            # feature_path = os.path.join(data_path,self.test_name[i]+'.npy')
            # feature = np.load(feature_path)
            # feature = feature.reshape((feature.shape[0],feature.shape[2])).T
            # self.test_data.append(ac(feature))

        self.train_label = np.fromfunction(lambda i,j:j==self.train_label[i],(len(self.train_label),max(self.train_label)+1),dtype=int)+0
        self.test_label = np.fromfunction(lambda i,j:j==self.test_label[i],(len(self.test_label),max(self.test_label)+1),dtype=int)+0
        print 'loading done'






