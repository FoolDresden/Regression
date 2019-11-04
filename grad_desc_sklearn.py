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


lin_reg_mod  = linear_model.LinearRegression()
lin_reg_mod.fit(X_train, y_train)
pred = lin_reg_mod.predict(X_test)
test_set_rmse = (np.sqrt(mean_squared_error(y_test, pred)))
test_set_r2 = sklearn.metrics.r2_score(y_test, pred)
print(test_set_rmse)
print(test_set_r2)
print(lin_reg_mod.score(X,Y))
print(lin_reg_mod.coef_)
print(lin_reg_mod.intercept_)



