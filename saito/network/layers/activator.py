import numpy as np

class Activator():
    def __init__(self, func_name=None):
        self.func_name = func_name

    def activate(self, x, func_name=None):
        if func_name == None:
            func_name = self.func_name
            if func_name == None:
                error = ''' please activate with
                Activator.activate(x, 'func_name')
                '''
                raise NameError(error)
        y = 0
        if func_name == 'sigmoid':
            y = 1.0 / (1+np.exp(-x))
        elif func_name == 'tanh':
            y = (np.tanh(x)+1)/2.0
        elif func_name == 'arctan':
            y = np.arctan(x) / np.pi + 1/2.0
        elif func_name == 'softmax':
            y = np.exp(x)/sum(np.exp(x))
        elif func_name == 'relu':
            y = np.maximum(x,0)
        else:
            error = '''
            it's an unsupported function

            supported function : 
            sigmoid
            tanh
            relu
            '''           
            raise NameError(error)
        return y

    def deactivate(self, x, func_name=None):
        if func_name == None:
            func_name = self.func_name
            if func_name == None:
                error = ''' please deactivate with
                Activator.deactivate(x, 'func_name')
                '''
                raise NameError(error)
        y = 0
        if func_name == 'sigmoid':
            y = x * (1 - x)
        elif func_name == 'tanh':
            y = (1-np.power(x,2))/2.0
        elif func_name == 'arctan':
            y = 1/(1+np.power(np.tan(np.pi*(x-1/2.0)),2))/np.pi
        elif func_name == 'relu':
            y = np.sign(x)
        else:
            error = '''
            it's an unsupported function

            supported function : 
            sigmoid
            tanh
            relu
            '''           
            raise NameError(error)
        return y
