import numpy as np
import sys
sys.path.append("../saito/")
import toolbox

class FCLayer:
    def __init__(self, h, W,mode):
        ## number of hidden element ##
        self.h = h
        self.W = W
        self.mode = mode
        self.inp = None
        if mode == "Output":
            self.activate = lambda x: np.exp(x) / np.sum(np.exp(x))
        elif mode == "Hidden":
            self.activate = np.tanh
        self.activate_div = lambda x: 1-np.tanh(x)**2

    def CalcForward(self, inp):
        self.inp = inp
        self.a = np.dot(self.W, self.inp)
        return self.activate(self.a)


    def CalcBackward(self, prev_delta):
        self.delta = prev_delta
        delta = self.activate_div(self.inp) * np.dot(self.W.T, prev_delta)
        return delta

    def Update(self, eta):
        self.div = np.dot(np.matrix(self.delta).T, np.matrix(self.inp))
        self.W = np.array(self.W - eta * self.div)
        if np.isnan(self.W).any():
            raise ValueError

    def __str__(self):
        return "the number of element:%d\nmode:%s\nW:%s"%(self.h,self.mode,self.W)
    
class NeuralNetwork:
    def __init__(self, setting):
        ## read network setting ##
        layer_settings = setting["layer_settings"]
        self.layer_num = setting["layer_num"]
        self.eta = setting["eta"]
        ## create layer instance and store in list ##
        self.layer_list = []
        for i in xrange(self.layer_num-1):
            h = layer_settings[i]
            h_next = layer_settings[i+1]
            W = np.random.randint(-100, 100, (h_next, h)) / 100.0

            if i == self.layer_num-2:
                mode = "Output"
            else:
                mode = "Hidden"
            layer = FCLayer(h, W, mode)
            self.layer_list.append(layer)

    def train(self, x_train, labels):
        ## x_train:train samples (datanum,featuredim), labels:label (datanum,classnum), eta:training coefficiend ##
        datanum = x_train.shape[0]
        iteration = 1
        for j in xrange(iteration):
            for i in xrange(datanum):
                print "data: %d/%d"%(i, datanum)
                x = x_train[i, :]
                t = labels[i, :]

                #print "\nForwardPropagetion..."
                output = self.ForwardPropagate(x)
                #print "\nBackPropagetion..."
                self.BackPropagate(t, output)

                #print "\nUpdate parameters..."
                self.Update()

    def predict(self, x_test, classnum):
        datanum = x_test.shape[0]
        result = np.zeros((datanum, classnum))
        for i in xrange(datanum):
            x = x_test[i, :]
            #print "\nPredicting..."
            output = self.ForwardPropagate(x)
            result[i, np.argmax(output)] = 1
        return result
    
    def ForwardPropagate(self, x):
        input = x
        for i,l in enumerate(self.layer_list):
            #print "%dth Layer calculating..."%(i+1)
            output = l.CalcForward(input)
            input = output
        return output
    
    def BackPropagate(self, t, output):
        prev_delta = output-t
        for i, l in enumerate(self.layer_list[::-1]):
            #print "%dth Layer calculating..."%(self.layer_num-i-1)
            delta = l.CalcBackward(prev_delta)
            prev_delta = delta
            
    def Update(self):
        for i,l in enumerate(self.layer_list):
            #print "%dth Layer updating..."%(i+1)
            delta = l.Update(self.eta)
           
    def __str__(self):
        ## show network information ##
        return "\n".join(map(str, self_list))

def main():
    ## load dataset and preprocess ##
    from sklearn.datasets import load_digits, fetch_mldata
    from sklearn import preprocessing
    from sklearn.metrics import classification_report
    from sklearn.cross_validation import train_test_split
    dataset = load_digits()
    dataset = fetch_mldata('MNIST original', data_home='.')

    data = preprocessing.normalize(dataset.data, norm='l2')
    lb = preprocessing.LabelBinarizer()
    label = lb.fit_transform(dataset.target)

#    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.10)
    train_data, train_label, test_data, test_label = toolbox.load_mnist()
    train_data = np.hstack((np.ones((train_data.shape[0], 1)), train_data))
    test_data = np.hstack((np.ones((test_data.shape[0], 1)), test_data))

    inp_dim = train_data.shape[1]
    label_dim = label.shape[1]
    print label

    ## network setting ##
    setting = {"layer_settings":[inp_dim,128,label_dim],
             "layer_num":3,
             "eta":0.05}

    ## create Network instance ##
    neuralnet = NeuralNetwork(setting)

    ## train Network ##
    neuralnet.train(train_data, train_label)

    ## predict ##
    result = NN.predict(test_data, label_dim)
    print classification_report(test_label, result)
if __name__ == "__main__":
    main()
