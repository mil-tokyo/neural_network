import numpy as np
import convlayer
x=np.random.normal(0,1,(10,25,25))
cl=convlayer.ConvLayer(10,20,(3,3),(1,1),'Grad')
cw=cl.weight

