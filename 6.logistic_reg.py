import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix

churn_df=pd.read_csv('ChurnData.csv')
#print(churn_df.head())

churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]

churn_df['churn']=churn_df['churn'].astype(int)
#print(churn_df.head(5))

x=np.asarray( churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']])
#print(x[0:5])
y=np.asarray(churn_df['churn'])
#print(y[0:5])

X=preprocessing.StandardScaler().fit(x).transform(x)
#print(X[0:5])

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=4)

LR=LogisticRegression(C=0.01,solver='liblinear').fit(X,y)

PredLR=LR.predict(x_test)


#print(metrics.accuracy_score(PredLR,y_test)*100)

yhat_prob=LR.predict_proba(x_test)
#print(yhat_prob)

######jaccard_index####################
from sklearn.metrics import jaccard_score
print(jaccard_score(y_test,PredLR,pos_label=0))
