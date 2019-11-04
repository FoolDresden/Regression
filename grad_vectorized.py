import math as math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


dataset=pd.read_csv('D:\shrey\Documents\StudyMaterial\\fods_project\Data.csv')
X1 = dataset.iloc[:,1].values
X2 = dataset.iloc[:,2].values
Y = dataset.iloc[:,3].values
X = dataset.iloc[:,1:3].values

#677.8960442352434, 4.42942799, -12.24177671
indices = list(range(X1.shape[0]));
num_inst = int(0.7*X1.shape[0]);
np.random.shuffle(indices)
train_ind = indices[:num_inst]
test_ind = indices[num_inst:]
interArray = np.ones([num_inst, 1])
interArray2 = np.ones([X1.shape[0]-num_inst, 1])


X1train, X1test = np.array(X1[train_ind]), np.array(X1[test_ind])
X2train, X2test = np.array(X2[train_ind]), np.array(X2[test_ind])
Ytrain, Ytest = np.array(Y[train_ind]), np.array(Y[test_ind])
Xtrain, Xtest = np.array(X[train_ind]), np.array(X[test_ind])
Xtrain, Xtest = np.concatenate((interArray, Xtrain),axis=1), np.concatenate((interArray2, Xtest),axis=1)


def calc_sum(w, n=3):
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

def mean(y):
    tot = 0
    mean = 0
    for i in range(y.shape[0]):
        mean = mean + y[i]
    mean = mean/y.shape[0]
    return mean

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
w=np.array([1, 1, 1])
thresh = 1e-11
cnt = 0
errplot = []

while cnt<=100:
    if cnt%20==0:
        err = mse(Xtrain[:,1], Xtrain[:,2], Ytrain, w)
        err = (2*err)/Xtrain.shape[0]
        err = math.sqrt(err)
        errplot.append(err)
    print(cnt)
    w = w - lr*Xtrain.transpose()@(Xtrain@w-Ytrain)
    cnt=cnt+1

pred_values = Xtest@w
sse = mse(Xtest[:,1], Xtest[:,2], Ytest, w, 3)
sse = 2*sse
sst = mst(Ytest)
rsq = (sst-sse)/sst
mserr = sse/Ytest.shape[0]
rmserr = math.sqrt(mserr)
 


