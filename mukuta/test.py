import numpy as np
import convnetwork
x=np.random.normal(0,1,(3,256,256))
cl=convnetwork.ConvNetwork([(3,256,256),[[(100,5,5),(1,1),(2,2)],[(120,3,3),(1,1),(2,2)]],[128,10]],'ReLU','Grad')

