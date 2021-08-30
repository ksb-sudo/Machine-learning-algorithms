import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics

my_data=pd.read_csv('drug200.csv')
new_data=pd.read_csv('new_file.csv')
#print(my_data.head(5))

X=my_data[['Age','Sex','BP','Cholesterol','Na_to_K']].values

#print(X[0:5])

le_sex=preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1]=le_sex.transform(X[:,1])


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3])


y=my_data['Drug']

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=3)

#print('Shape of X training set {}'.format(x_train.shape),'&',' Size of Y training set {}'.format(y_train.shape))
#print(x_test.shape,y_test.shape)

drugTree=DecisionTreeClassifier(criterion='entropy',max_depth=4)

drugTree.fit(x_train,y_train)

predTree=drugTree.predict(x_test)
#print(predTree[0:5])
#print(y_test[0:5])

#print(metrics.accuracy_score(predTree,y_test)*100)

##################################################################
#################predicting with new values######################
x=new_data[['Age','Sex','BP','Cholesterol','Na_to_K']].values

sex=preprocessing.LabelEncoder()
sex.fit(['F','M'])
x[:,1]=sex.transform(x[:,1])


BP = preprocessing.LabelEncoder()
BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
x[:,2] = BP.transform(x[:,2])


Chol = preprocessing.LabelEncoder()
Chol.fit([ 'NORMAL', 'HIGH'])
x[:,3] = Chol.transform(x[:,3])

ptree=drugTree.predict(x)
print(ptree)