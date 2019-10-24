import math as math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


dataset=pd.read_csv(r"D:\shrey\Documents\StudyMaterial\\fods_project\Data.csv")
X1 = dataset.iloc[:,1].values
X2 = dataset.iloc[:,2].values
Y = dataset.iloc[:,3].values

indices = list(range(X1.shape[0]));
num_inst = int(0.7*X1.shape[0]);
np.random.shuffle(indices)
train_ind = indices[:num_inst]
test_ind = indices[num_inst:]

X1train, X1test = X1[train_ind], X1[test_ind]
X2train, X2test = X2[train_ind], X2[test_ind]
Ytrain, Ytest = Y[train_ind], Y[test_ind]


def calc_sum(w, num, n=3):
    new_w=np.zeros(n)
    i = np.random.randint(0,num)
    new_w[0]= (w[0]+w[1]*X1train[i]+w[2]*X2train[i]-Ytrain[i])
    new_w[1]=X1train[i]*(w[0]+w[1]*X1train[i]+w[2]*X2train[i]-Ytrain[i])
    new_w[2]= X2train[i]*(w[0]+w[1]*X1train[i]+w[2]*X2train[i]-Ytrain[i])
    return new_w

def mse(x1, x2, y, w, n=3):
    err = 0;
    #print(x1.shape[0])
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
    
lr = 1e-6;
w = np.ones(3);
thresh = 1e-6
cnt = 0
init_test_err = (mse(X1test, X2test, Ytest, w, 3))
init_train_err = mse(X1train, X2train, Ytrain, w, 3)
errplot = []

while cnt<=150:
    print(cnt)
    for i in range(num_inst):
        new_n = calc_sum(w,X1train.shape[0],3);
        a = lr*new_n[0]
        b = lr*new_n[1]
        c = lr*new_n[2]
        w = w - [a, b, c]
        if abs(a)<=thresh and abs(b)<=thresh and abs(c)<=thresh:
            break
        err = mse(X1train, X2train, Ytrain, w, 3)
        mserr = 2*err/X1train.shape[0]
        rmserr = math.sqrt(mserr)
        errplot.append(rmserr)
        print (w)
        print(a)
        print(b)
        print(c)
        print(rmserr)
        print("\n")
    cnt=cnt+1        


pred_values = w[0] + w[1]*X1 + w[2]*X2
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
ax.scatter3D(X1,X2,pred_values,c="yellow")
plt.show()
'''