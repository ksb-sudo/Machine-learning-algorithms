import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score


df=pd.read_csv('Fuel consumption 2019.csv')
#print(df.head())

#print(df.describe())

cdf=df[['Engine_Size','Cylinders','Fuel_Consumption_Comb','CO2_Emissions']]

print(cdf.head(9))

viz=cdf[['Engine_Size','Cylinders','Fuel_Consumption_Comb','CO2_Emissions']]
#viz.hist()
#plt.show()

'''plt.scatter(cdf.Fuel_Consumption_Comb,cdf.CO2_Emissions,color='blue')
plt.xlabel('Fuel-Consumption-Comb')
plt.ylabel('CO2-Emissions')
plt.show()

plt.scatter(cdf.Engine_Size,cdf.CO2_Emissions,color='blue')
plt.xlabel('Engine_Size')
plt.ylabel('CO2-Emissions')
plt.show()'''

msk=np.random.rand(len(df))<0.8
train=cdf[msk]
test=cdf[~msk]

plt.scatter(train.Engine_Size,train.CO2_Emissions,color='blue')
plt.xlabel('Engine_Size')
plt.ylabel('CO2-Emissions')
#plt.show()

regr=linear_model.LinearRegression()
train_x=np.asanyarray(train[['Engine_Size']])
train_y=np.asanyarray(train[['CO2_Emissions']])
regr.fit(train_x,train_y)

plt.scatter(train.Engine_Size,train.CO2_Emissions,color='blue')
plt.plot(train_x,regr.coef_[0][0]*train_x,regr.intercept_[0],'-r')
plt.xlabel('Engine-size')
plt.ylabel('Emission')
plt.show()


test_x=np.asanyarray(test[['Engine_Size']])
test_y=np.asanyarray(test[['CO2_Emissions']])
_y_=regr.predict(test_x)

mean=np.mean(np.absolute(_y_-test_y))
mean_sq=np.mean((_y_-test_y)**2)
r_2=r2_score(test_y,_y_)
print(f'The MAE IS: {mean}')
print(f'The MSE IS: {mean_sq}')
print(f'The R-SQUARE IS: {r_2}')

#####################################################################
