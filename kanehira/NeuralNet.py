import numpy as np
import math

class FCLayer:
    
    def __init__(self,h,W,mode):
        ## number of hidden element ##
        self.h=h
        self.W=W
        self.mode=mode
        if mode == "Output":
            self.activate=lambda x:np.exp(x) / np.sum(np.exp(x)) 
        elif mode == "Hidden":
            self.activate=np.tanh
        self.activate_div=lambda x:(1-np.tanh(x)**2)

    def CalcForward(self,inp):
        self.inp=inp
        self.a=np.dot(self.W,self.inp)
        return self.activate(self.a)


    def CalcBackward(self,prev_delta):
        self.delta=prev_delta
        delta=self.activate_div(self.inp) * np.dot(self.W.T,prev_delta)
        return delta

    def Update(self,eta):
        self.div=np.dot(np.matrix(self.delta).T,np.matrix(self.inp))
        self.W = np.array(self.W - eta * self.div)


    def __str__(self):
        return "the number of element:%d\nmode:%s\n"%(self.h,self.mode)

        
class NeuralNetwork:
    
    def __init__(self,setting):
        ## read network setting ##
        layer_settings=setting["layer_settings"]
        self.layer_num=setting["layer_num"]
        
        ## create layer instance and store in list ##
        self.layer_list=[]
        for i in xrange(self.layer_num-1):
            h=layer_settings[i]
            h_next=layer_settings[i+1]
            W = np.random.randint(-100,100,(h_next,h)) / 100.0

            if i==self.layer_num-2:
                mode="Output"
            else:
                mode="Hidden"
                
            layer=FCLayer(h,W,mode)
            self.layer_list.append(layer)

        for i,l in enumerate(self.layer_list):
            print "%dth Layer information"%(i+1)
            print l
            

    def train(self,x_train,labels,eta):
        ## x_train:train samples (datanum,featuredim), labels:label (datanum,classnum), eta:training coefficiend ##
        datanum=x_train.shape[0]
        for i in xrange(datanum):
            x=x_train[i,:]
            t=labels[i,:]

            print x
            print "ForwardPropagetion...\n"
            output=self.ForwardPropagate(x)
        
            print "BackPropagetion...\n"
            self.BackPropagate(t,output)

            print "Update parameters...\n"
            self.Update(eta)

    def predict(self,x_test,classnum):
        datanum=x_test.shape[0]
        result=np.zeros((datanum,classnum))
        for i in xrange(datanum):
            x=x_test[i,:]
            print "Predictiong...\n"
            result[i,:]=self.ForwardPropagate(x)
        return result
        
    def ForwardPropagate(self,x):
        input=x
        for i,l in enumerate(self.layer_list):
            print "%dth Layer calculating...\n"%(i+1)
            output=l.CalcForward(input)
            input=output
            print "inp"
            print input

        return output
    
    def BackPropagate(self,t,output):
        prev_delta=t-output
        for i,l in enumerate(self.layer_list[::-1]):
            print "%dth Layer calculating...\n"%(self.layer_num-i-1)
            delta=l.CalcBackward(prev_delta)
            prev_delta=delta
            
    def Update(self,eta):
        for i,l in enumerate(self.layer_list):
            print "%dth Layer updating...\n"%(i+1)
            delta=l.Update(eta)

        
if __name__ == "__main__":
    setting={"layer_settings":[10,3,5],
             "layer_num":3}
    
    NN=NeuralNetwork(setting)
    x=np.ones((10,10))
    labels=np.zeros((10,5))
    eta=0.1
    NN.train(x,labels,eta)
    x_test=np.ones((10,10))
    result=NN.predict(x_test,5)
    print result
    
