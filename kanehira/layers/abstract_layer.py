from abc import ABCMeta, abstractmethod

class Layer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward_calculate(self, inp):
        pass

    @abstractmethod
    def back_calculate(self, prev_delta):
        pass

    @abstractmethod
    def update(self, eta, batch_size):
        pass

    def get_params(self, param_name):
        return getattr(self, param_name)

    def set_params(self, param_name, value):
        setattr(self, param_name, value)

    
