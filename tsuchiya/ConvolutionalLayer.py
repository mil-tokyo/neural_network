import numpy as np
from Layer import Layer

class ConvolutionalLayer(Layer):
	_window = (5, 5)

	def __init__(self, _shape, _lr = 0.1, window=(5, 5), stride=1):
		super(ConvolutionalLayer, self).__init__(_shape, _lr)
		self._window = window

	def append(self, _child, window=(5, 5)):
		self._child = _child
		self._child.setParent(self)

		if self._child.type() == 'convolutional':
			self._initWeights((self._units.shape[0], self._child.getUnits().shape[0], self._child.window()[0], self._child.window()[1]))

#		self._initWeights((self._units.shape[0], self._units.shape[1], self._window[0], self._window[1]))
		return self._child

	def forward(self, _units):
		self._units = _units
		self._outputs = self.activate(self._units)

		if (not self._child == None):
			if self._child.type() == 'convolutional':
				next_units = Layer.convConvForward(self._weights, self._units, self._child.getUnits().shape)
				self._child.forward(next_units)

			elif self._child.type() == 'pooling':
				self._child.forward(self._units)

	def backward(self, _prev_diff):
		pass

	def window(self):
		return self._window

	def type(self):
		return 'convolutional'
