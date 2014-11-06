import numpy as np

class Loss():
    def __init__(self, func_name=None):
        self.func_name = func_name

    def loss(self, out, label, func_name=None):
        if func_name == None:
            func_name = self.func_name
            if func_name == None:
                error = ''' please activate with
                Loss.loss(x, 'func_name')
                '''
                raise NameError(error)

        label_array = np.zeros(len(out))
        label_array[label] += 1

        if func_name == 'square':
            self.grad = out - label_array
            self.error = np.dot(self.grad, self.grad)/2
        else:
            error = '''
            it's an unsupported function

            supported function :
            sigmoid
            tanh
            relu
            '''
            raise NameError(error)