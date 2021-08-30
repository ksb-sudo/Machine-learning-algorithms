import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

df=pd.read_csv('teleCust1000t.csv')
#print(df.head())

#print(df['custcat'].value_counts())
data=df.hist(column='income',bins=50)
#print(data)
#plt.show()

X=df[['region','tenure','age','marital','address','income','ed','employ','retire','gender','reside']].values.astype(float)

y=df['custcat'].values

X=preprocessing.StandardScaler().fit(X).transform(X.astype(float))
#print(X[0:5])

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=4)
#print("train set",x_train.shape,y_train.shape)
#print("test set",x_test.shape,y_test.shape)

k=9
negh=KNeighborsClassifier(n_neighbors=k).fit(x_train,y_train)

yhat=negh.predict(x_test)

print('Train accuracy',metrics.accuracy_score(y_train,negh.predict(x_train)))
print('test accuracy',metrics.accuracy_score(y_test,yhat))