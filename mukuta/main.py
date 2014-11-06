import sys
sys.path.append('../saito')
import toolbox
import numpy as np
import convnetwork
cl=convnetwork.ConvNetwork([(1,28,28),[[(10,5,5),(1,1),(2,2)],[(12,3,3),(1,1),(2,2)]],[128,10]],'ReLU','Grad')

[train_images, train_labels, test_images, test_labels] = toolbox.load_mnist()
