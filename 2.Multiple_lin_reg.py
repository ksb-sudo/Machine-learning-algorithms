import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn import linear_model

df=pd.read_csv('Fuel consumption 2019.csv')

cdf=df[['Engine_Size','Cylinders','Fuel_Consumption_City','Fuel_Consumption_Hwy','Fuel_Consumption_Comb','CO2_Emissions']]

#print(cdf.head(9))

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


regr=linear_model.LinearRegression()
x=np.asanyarray(train[['Engine_Size','Cylinders','Fuel_Consumption_Comb']])
y=np.asanyarray(train[['CO2_Emissions']])
regr.fit(x,y)

y_hat=regr.predict(test[['Engine_Size','Cylinders','Fuel_Consumption_Comb']])
x=np.asanyarray(test[['Engine_Size','Cylinders','Fuel_Consumption_Comb']])
y=np.asanyarray(test[['CO2_Emissions']])

print(len(x),len(y))

print(regr.score(x,y))