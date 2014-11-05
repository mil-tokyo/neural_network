from layers.convolutional_layer import ConvolutionalLayer
from layers.maxpooling_layer import MaxPooling
from layers.activation_layer import Activation 

import sys
sys.path.append("../saito/")
import toolbox
import numpy as np

if __name__ == "__main__":
    input_kernel_size = 30
    input_shape = (30, 30)
    output_kernel_size = 10
    output_shape = (26, 26)
    input_row, input_col = input_shape
    output_row, output_col = output_shape
    
    network_setting = {"input_kernel_size" : input_kernel_size,
                       "input_shape" : input_shape,
                       "output_kernel_size" : output_kernel_size,
                       "output_shape" : output_shape,
                       "window_size" : 5,
                       "step_size" : 1
                       }

    models = [ConvlutionalLayer(network_setting), MaxPooling(), Activation()]
    
    X = np.ones((input_kernel_size, input_row, input_col))
    t = np.ones((output_kernel_size, output_row / 2, output_col / 2))
    
    output = X
    for m in models:
        output = m.forward_calculate(output)
        
    prev_delta = output - t
    for m in models[::-1]:
        prev_delta = m.back_calculation(prev_delta)

    for m in models:
        m.update()
       
    print output



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
