import math as math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


dataset=pd.read_csv('Dataset.txt')
X1 = dataset.iloc[:,1].values
X2 = dataset.iloc[:,2].values
Y = dataset.iloc[:,3].values

#677.8960442352434, 4.42942799, -12.24177671
indices = list(range(X1.shape[0]));
num_inst = int(0.7*X1.shape[0]);
np.random.shuffle(indices)
train_ind = indices[:num_inst]
test_ind = indices[num_inst:]

X1train, X1test = X1[train_ind], X1[test_ind]
X2train, X2test = X2[train_ind], X2[test_ind]
Ytrain, Ytest = Y[train_ind], Y[test_ind]

def calc_sum(w, n=3):
    #n = 3
    sum = np.zeros(n)
    for i in range(num_inst):
        sum[0] =sum[0] + (w[0]+w[1]*X1train[i]+w[2]*X2train[i]-Ytrain[i])
        sum[1] =sum[1] + X1train[i]*(w[0]+w[1]*X1train[i]+w[2]*X2train[i]-Ytrain[i])
        sum[2] =sum[2] + X2train[i]*(w[0]+w[1]*X1train[i]+w[2]*X2train[i]-Ytrain[i])
    return sum

def mse(x1, x2, y, w, n=3):
    err = 0;
    print(x1.shape[0])
    for i in range(x1.shape[0]):
        err = err + (0.5)*(((w[0]+w[1]*x1[i]+w[2]*x2[i]) - y[i])**2)
    return err    

def mst(y):
    tot = 0
    mean = 0
    for i in range(y.shape[0]):
        mean = mean + y[i]
    mean = mean/y.shape[0]
    for i in range(y.shape[0]):
        tot = tot + (y[i]-mean)**2
    return tot

lr = 1e-9;
#w = np.array([677.8960442352434, 4.42942799, -12.24177671])
#w = np.array([0.95500818, 0.95106313, 0.20977437])
w=np.array([1, 1, 1])
thresh = 1e-11
cnt = 0
init_test_err = (mse(X1test, X2test, Ytest, w, 3))
init_train_err = mse(X1train, X2train, Ytrain, w, 3)
errplot = []

while cnt<=100:
    print(cnt)
    sum = calc_sum(w,3);
    a = lr*sum[0]
    b = lr*sum[1]
    c = lr*sum[2]
    #print(sum)
    print(w)
    print(a)
    print(b)
    print(c)
    if abs(a)<=thresh and abs(b)<=thresh and abs(c)<=thresh:
        break
    w = w - [a, b, c]
    err = mse(X1train, X2train, Ytrain, w, 3)
    errplot.append(err)
    print(math.sqrt(err/X1train.shape[0]))
    print("\n")
    cnt=cnt+1


pred_values = w[0] + w[1]*X1test + w[2]*X2test
sse = mse(X1test, X2test, Ytest, w, 3)
sse = 2*sse
sst = mst(Ytest)
rsq = (sst-sse)/sst
mserr = sse/Ytest.shape[0]
rmserr = math.sqrt(mserr)
 
'''
fig = plt.figure()
ax = plt.axes(projection="3d")
#ax.scatter3D(X1,X2,Y,c="red")
ax.scatter3D(X1test,X2test,pred_values,c="yellow")
ax.view_init(45, 0)
plt.show()
'''


