'''lm = linear_model.LinearRegression()
model = lm.fit(X,y)'''

import math as math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
#from sklearn.metrics import r2_Score

dataset=pd.read_csv('D:\shrey\Documents\StudyMaterial\\fods_project\Data.csv')
X1 = dataset.iloc[:,1].values
X2 = dataset.iloc[:,2].values
X=dataset.iloc[:,1:3].values
Y = dataset.iloc[:,3].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=9)



'''
indices = list(range(X1.shape[0]));
num_inst = int(0.7*X1.shape[0]);
np.random.shuffle(indices)
train_ind = indices[:num_inst]
test_ind = indices[num_inst:]

X1train, X1test = X1[train_ind], X1[test_ind]
X2train, X2test = X2[train_ind], X2[test_ind]
Ytrain, Ytest = Y[train_ind], Y[test_ind]
'''



lin_reg_mod  = linear_model.LinearRegression()
lin_reg_mod.fit(X_train, y_train)
pred = lin_reg_mod.predict(X_test)
test_set_rmse = (np.sqrt(mean_squared_error(y_test, pred)))
test_set_r2 = sklearn.metrics.r2_score(y_test, pred)
print(test_set_rmse)
print(test_set_r2)
print(lin_reg_mod.score(X,Y))
'''
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
w = np.array([0.95487231, 0.94850383, 0.21021318])
thresh = 1e-11
cnt = 0
init_test_err = (mse(X1test, X2test, Ytest, w, 3))
init_train_err = mse(X1train, X2train, Ytrain, w, 3)
errplot = []

while True:
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


pred_values = w[0] + w[1]*X1 + w[2]*X2
sse = mse(X1, X2, Y, w, 3)
sse = 2*sse
sst = mst(Y)
rsq = (sst-sse)/sst
mserr = sse/Y.shape[0]
rmserr = math.sqrt(mserr)
 
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(X1,X2,Y,c="red")
ax.scatter3D(X1,X2,pred_values,c="yellow")
ax.view_init(45, 0)
plt.show()
'''



