import numpy as np
import convnetwork
x=np.random.normal(0,1,(1,28,28))
cl=convnetwork.ConvNetwork([(1,28,28),[[(10,5,5),(1,1),(2,2)],[(12,3,3),(1,1),(2,2)]],[128,10]],'ReLU','Grad')

