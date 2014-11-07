class Net:
    def __init__(self):
        self._first_layers = []
        self._last_added_layer = None

    def addLayer(self, _layer):
        if (self._last_added_layer == None):
            self._first_layers.append(_layer)
            self._last_added_layer = _layer
        else:
            self._last_added_layer.append(_layer)
