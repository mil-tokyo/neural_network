import numpy as np
from Layer import Layer

class PoolingLayer(Layer):
	_window = (5, 5)
	def __init__(self, _shape, window=(2,2), overlap=0):
		super(PoolingLayer, self).__init__(_shape)
		self._window = window

	def append(self, _child, window=(2, 2)):
		self._child = _child
		self._child.setParent(self)

		if self._child.type() == 'convolutional':
			self._initWeights((self._units.shape[0], self._child.getUnits().shape[0], self._child.window()[0], self._child.window()[1]))
		elif self._child.type() == 'fullyconnected':
			self._initWeights((self._units.size, self._child.getLength()))

		return self._child

	def forward(self, _units):
		self._units, self._pooled_inds = self.maxPool(_units)
		self._outputs = self.activate(self._units)

		if (not self._child == None):
			if self._child.type() == 'convolutional':
				next_units = Layer.convConvForward(self._weights, self._outputs, self._child.getUnits().shape)
				self._child.forward(next_units)
			elif self._child.type() == 'fullyconnected':
				self._child.forward(np.dot(self._outputs.flatten(), self._weights) + self._bias)

	def maxPool(self, _inputs):
		pooled = np.zeros((_inputs.shape[0], _inputs.shape[1]/self._window[0], _inputs.shape[2]/self._window[1]))
		pooled_inds = np.zeros((_inputs.shape[0], _inputs.shape[1]/self._window[0], _inputs.shape[2]/self._window[1]), dtype=int)
		for i in range(0, pooled.shape[1]):
			for j in range(0, pooled.shape[2]):
				x = i*self._window[0]; x_ = x+self._window[0]
				y = j*self._window[1]; y_ = y+self._window[1]
				for k in range(0, pooled.shape[0]):
					pooled[k,i,j] = np.max(_inputs[k,x:x_, y:y_])
					pooled_inds[k,i,j] = np.argmax(_inputs[k,x:x_, y:y_])
		return pooled, pooled_inds

	def backward(self, _prev_diff):
		if (not self._parent == None):
			self._calcDiff(_prev_diff)
			unpooled_diff = np.zeros((self._units.shape[0], self._units.shape[1]*self._window[0], self._units.shape[2]*self._window[1]))
			for i in range(0, self._units.shape[1]):
				for j in range(0, self._units.shape[2]):
					x = i*self._window[0]; x_ = x+self._window[0]
					y = j*self._window[1]; y_ = y+self._window[1]
					for k in range(0, self._units.shape[0]):
						submat = np.zeros(self._window)
						ind = np.unravel_index(self._pooled_inds[k,i,j], self._window)
						submat[ind[0],ind[1]] = self._diff[k,i,j]
						unpooled_diff[k,x:x_,y:y_] = submat


			self._parent.backward(unpooled_diff)

	def _calcDiff(self, _prev_diff):
		if self._child.type() == 'fullyconnected':
			self._diff = self._activation_func_diff(self._units.flatten()) * np.dot(self._weights, _prev_diff)
			self._diff = self._diff.reshape(self._units.shape)


	def type(self):
		return 'pooling'