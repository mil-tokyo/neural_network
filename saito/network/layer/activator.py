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
            y = np.tanh(x)
        elif func_name == 'relu':
            if np.isscalar(x):
                y = float(max(0,x))
            else:
                y = x
                for i in range(0,len(x)):
                    y[i] = max(0,y[i])
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
            y = 1/(1+np.power(np.tan(x),2))
        elif func_name == 'relu':
            if np.isscalar(x):
                y = float(max(0,x))
            else:
                y = x
                for i in range(0,len(x)):
                    y[i] = max(0,y[i])
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
