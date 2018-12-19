import numpy as np 
#import matplotlib.pyplot as pt 
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


data=pd.read_csv("train.csv").as_matrix()

#training dataset
xtrain=data[0:21000,1:]
train_label=data[0:21000,0]

clf=KNeighborsClassifier() #default value of k is 5
clf.fit(xtrain,train_label)

#test dataset
xtest=data[21000:,1:]
actual_label=data[21000:,0]


#Accuracy calculation
p=clf.predict(xtest)
count=0
for i in range(0,21000):
	count+=1 if p[i]==actual_label[i] else 0
print('K Nearest Neighbours(k=5)\n')
print('Accuracy: ',count*100/21000)

