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

dataset=pd.read_csv('D:\shrey\Documents\StudyMaterial\\fods_project\Data.csv')
X1 = dataset.iloc[:,1].values
X2 = dataset.iloc[:,2].values
Y = dataset.iloc[:,3].values
X1 = (X1 - np.min(X1))/(np.max(X1)-np.min(X1))
X2 = (X2 - np.min(X2))/(np.max(X2)-np.min(X2))
Y = (Y - np.min(Y))/(np.max(Y)-np.min(Y))
X = dataset.iloc[:,1:3].values


#677.8960442352434, 4.42942799, -12.24177671
indices = list(range(X1.shape[0]));
num_inst = int(0.6*X1.shape[0]);
valid_inst = int(0.2*X1.shape[0]);
np.random.shuffle(indices)
train_ind = indices[:num_inst]
valid_ind = indices[num_inst:num_inst+valid_inst]
test_ind = indices[num_inst+valid_inst:]
interArray = np.ones([num_inst, 1])
interArray2 = np.ones([X1.shape[0]-num_inst-valid_inst, 1])


X1train, X1test = np.array(X1[train_ind]), np.array(X1[test_ind])
X2train, X2test = np.array(X2[train_ind]), np.array(X2[test_ind])
X1valid,  X2valid = np.array(X1[valid_ind]), np.array(X2[valid_ind])
X1valid,  X2valid = X1valid.reshape(X1valid.shape[0],1), X2valid.reshape(X2valid.shape[0],1)
Yvalid = np.array(Y[valid_ind])
Yvalid = Yvalid.reshape(Yvalid.shape[0],1)
X1train, X1test = X1train.reshape(X1train.shape[0],1), X1test.reshape(X1test.shape[0],1)
X2train, X2test = X2train.reshape(X2train.shape[0],1), X2test.reshape(X2test.shape[0],1)
Ytrain, Ytest = np.array(Y[train_ind]), np.array(Y[test_ind])
Ytrain, Ytest = Ytrain.reshape(Ytrain.shape[0],1), Ytest.reshape(Ytest.shape[0],1)
Xtrain=interArray
Xtest=interArray2
Xvalid = np.ones([valid_inst,1])
#Xtrain, Xtest = np.concatenate((interArray, Xtrain),axis=1), np.concatenate((interArray2, Xtest),axis=1)
n = 6


def makeDot(x, y):
    return x**y

def mse(x, y, w):
    err = 0
    print("calcing err")
    w=w.transpose()
    x=x.transpose()
    tot=w@x
    tot=tot.transpose()-y
    tot=(tot)**2
    err = np.sum(tot)
    '''for i in range(0, x.shape[0]):
        z = x[i].reshape(x[i].shape[0],1)
        err = err + ((np.sum(w.transpose()*z)-y[i]))**2'''
    print("calced err")
    return err


order = Xtrain.shape[1]
for i in range(1,n+1):
    for j in range(0,i+1):
        print('x',j,'; y',i-j)
        order = order + 1
        x = makeDot(X1train, j)*makeDot(X2train, i-j)
        y = makeDot(X1test, j)*makeDot(X2test, i-j)
        z = makeDot(X1valid, j)*makeDot(X2valid, i-j)
#        print(x)
        Xtrain = np.concatenate((Xtrain, x), axis=1)
        Xtest = np.concatenate((Xtest, y), axis=1)      
        Xvalid = np.concatenate((Xvalid,z), axis=1)      
       
print(order)
w = np.ones([order,1])

lr = 1e-8;
cnt = 0
errplot = []

lamdas = [1e-6, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
lamda_err = []
minerr = 1e20
w_ans = []
errplot_ans = []


for l in lamdas:
    while cnt<=10000:
        '''if cnt%20==0:
            err = mse(Xtrain, Ytrain, w)
            err = err/Xtrain.shape[0]
            err = math.sqrt(err)
            print(err)
            errplot.append(err)'''
        print(cnt)
        w = w - lr*Xtrain.transpose()@(Xtrain@w-Ytrain) -l*w
        cnt=cnt+1
    verr = mse(Xvalid, Yvalid, w)
    verr = (verr)/Xvalid.shape[0]
    verr = math.sqrt(verr)
    print(l)
    print(verr)
    print("\n")
    lamda_err.append(verr)
    if verr < minerr:
        w_ans = w
        lamda = l
        minerr = verr
        errplot_ans = errplot
    errplot = []
    #l = l + 1
    cnt = 0
    w=np.ones([order, 1])
    

predVal = ((w.transpose()@Xtest.transpose()).transpose()-Ytest)**2
sst = (Ytest-((np.sum(Ytest))/Ytest.shape[0]))**2
sst = np.sum(sst)
rsq = (sst - mse(Xtest, Ytest, w))/sst

lin_reg_mod  = linear_model.LinearRegression()
lin_reg_mod.fit(Xtrain,Ytrain)
pred = lin_reg_mod.predict(Xtest)
test_set_rmse = (np.sqrt(mean_squared_error(Ytest, pred)))
test_set_r2 = sklearn.metrics.r2_score(Ytest, pred)
print(test_set_rmse)
print(test_set_r2)
print(lin_reg_mod.coef_)
print(lin_reg_mod.intercept_)

'''
n=2
array([[ 0.2295455 ],
       [-0.14525967],
       [ 0.05349196],
       [ 0.04925907],
       [ 0.01513767],
       [ 0.01445871]])
    
n=6
array([[ 0.17956853],
       [ 0.02163912],
       [ 0.22639285],
       [-0.06258829],
       [-0.04743586],
       [-0.02321694],
       [-0.09658372],
       [-0.04320904],
       [-0.06424025],
       [-0.14323994],
       [-0.10579922],
       [-0.00633331],
       [ 0.0275315 ],
       [-0.01997516],
       [-0.15929745],
       [-0.09898844],
       [ 0.03184349],
       [ 0.10938688],
       [ 0.12328113],
       [ 0.05562727],
       [-0.11731754],
       [-0.08142594],
       [ 0.0682257 ],
       [ 0.17301099],
       [ 0.22845552],
       [ 0.22432027],
       [ 0.14186359],
       [-0.04865297]])    
'''    