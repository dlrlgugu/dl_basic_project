import time
import numpy as np
import pandas as pd
#import seaborn as sns
import h5py
import matplotlib.pyplot as plt
from dnn_app_utils_v3 import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

train = pd.read_csv(r'C:\gugu\project\dl_basic_project\titanic_with_dl\datasets\titanic_train.csv')

age = train['Age'].values.reshape(887,1)
scaler = StandardScaler()
scaler.fit(age)
scaled_age = scaler.transform(age)
train['Age'] = scaled_age


fare = train['Fare'].values.reshape(887,1)
scaler = StandardScaler()
scaler.fit(fare)
scaled_fare = scaler.transform(fare)
train['Fare'] = scaled_fare



X=train.drop('Survived',axis=1)
X=train.drop('Name',axis=1)
#X=train.drop('Sex',axis=1)
Y=train['Survived']


sex=X['Sex']
#print(X['Sex'][0])
#print(sex[0])
for i in range(0,len(sex)):
    if sex[i] == 'male':
        X['Sex'][i]=0
    else :
        X['Sex'][i]=1
#why this takes so forever??

#a=np.int64(train['Age'])
#X['Age'] = np.asarray(train['Age'],dtype=int)
#X['Fare'] = np.asarray(train['Fare'],dtype=int)

"""
age=train['Age']
for i in range(0,len(age)):
    train['Age'][i] = int(age[i])
"""

X_train ,  X_test , Y_train, Y_test = train_test_split(X,Y,test_size=0.2)


X_train = np.array(X_train,dtype=float)
X_test = np.array(X_test,dtype=float)
Y_train = np.array(Y_train,dtype=float)
Y_test = np.array(Y_test,dtype=float)

X_train = X_train.T
X_test = X_test.T
Y_train = Y_train.reshape(1,Y_train.shape[0])
Y_test = Y_test.reshape(1,Y_test.shape[0])

#print(type(X_train))

print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)

layers_dims = [7,16,32,64,128,16,1]
parameters = L_layer_model(X_train,Y_train,layers_dims,num_iterations = 2000, print_cost = True)

pred_train = predict(X_train, Y_train, parameters)
pred_test = predict(X_test, Y_test, parameters)


Question = np.array([[0,3.,1.,32.,1.,0.,7.25],
                    [0,2.,1.,32.,1.,0.,30.0708],
                    [0,1.,1.,32.,1.,0.,71.2833],
                     [0,3,0,30,1,0,7.25],
                     [0,2,0,30,1,0,30.0708],
                     [0,1,0,30,1,0,71.2833]],dtype=float)

Question = Question.T
probas, caches = L_model_forward(Question, parameters)
p = np.zeros((1,Question.shape[1]))

for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

print(p)


