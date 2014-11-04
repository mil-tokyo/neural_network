import sys
sys.path.append('../saito')
import toolbox
import numpy as np
import fcnetwork

[train_images, train_labels, test_images, test_labels] = toolbox.load_mnist()
trl=np.fromfunction(lambda i,j:j==train_labels[i],(train_labels.size,10),dtype=int)+0
