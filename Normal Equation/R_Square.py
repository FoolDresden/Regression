import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from Normal_Equations import *
from math import *


Ypred = np.zeros((test_size,), dtype = int )
for i in range(test_size):
	Ypred[i] = (W[0] + W[1]*X1test[i] + W[2]*X2test[i])
xs = 0L
xss = 0L
ys = 0L
yss = 0L
xys = 0L

for i in range(test_size):
	xss += Ypred[i]*Ypred[i]
	ys += Ytest[i]
	yss += Ytest[i]*Ytest[i]
	xs += Ypred[i]
	xys += Ytest[i]*Ypred[i]
den = sqrt((test_size*xss - xs**2)*(test_size*yss - ys**2))
num = test_size*xys - xs*ys
Rsq = num/den
print(xs/test_size)