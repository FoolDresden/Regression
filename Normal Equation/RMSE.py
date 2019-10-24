import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from Normal_Equations import *
from math import *


Ypred = np.zeros((test_size,), dtype = int )
for i in range(test_size):
	Ypred[i] = (W[0] + W[1]*X1test[i] + W[2]*X2test[i])
E = 0
for i in range (test_size):
	error = Ytest[i] - Ypred[i]
	errsq = error*error
	E += errsq
mse = E/test_size
rmse = sqrt(mse)	
print('RMSE =')
print (rmse)


