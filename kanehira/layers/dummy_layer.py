from abstract_layer import Layer

class DummyLayer(Layer):
    def __init__(self):
        pass

    def forward_calculate(self, inp):
        return 0

    def back_calculate(self, delta):
        pass

    def update(self, eta, batch_size):
        pass

