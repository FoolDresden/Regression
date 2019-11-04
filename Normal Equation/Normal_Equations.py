import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from numpy.linalg import inv

#Read Dataset variables X1, X2 & Y
dataset = pd.read_csv('D:\shrey\Documents\StudyMaterial\\fods_project\Data.csv')
X1 = dataset.iloc[:,1].values
X2 = dataset.iloc[:,2].values
Y = dataset.iloc[:,3].values


indices = list(range(X1.shape[0]));
num_inst = int(0.7*X1.shape[0]);
np.random.shuffle(indices)
train_ind = indices[:num_inst]
test_ind = indices[num_inst:]
train_size = len(train_ind)
test_size = len(test_ind)


X1train, X1test = X1[train_ind], X1[test_ind]
X2train, X2test = X2[train_ind], X2[test_ind]
Ytrain, Ytest = Y[train_ind], Y[test_ind]
X1train = np.asarray(X1train)
X1test = np.asarray(X1test)
X2train = np.asarray(X2train)
X2test = np.asarray(X2test)
Ytest = np.asarray(Ytest)
Ytrain = np.asarray(Ytrain)
X0 = np.ones((train_size,), dtype=int)

X = [X0,X1train,X2train]
X = np.asarray(X)
Xt = X.transpose()
Xdot = np.dot(X, Xt)
Xinv = inv(Xdot)
Xy = np.dot(X, Ytrain)
W = np.dot(Xinv, Xy)
print(W)



