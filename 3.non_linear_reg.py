import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import r2_score

df=pd.read_csv('Fuel consumption 2019.csv')

cdf=df[['Engine_Size','Cylinders','Fuel_Consumption_Comb','CO2_Emissions']]

#print(cdf.head(9))

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

train_x = np.asanyarray(train[['Engine_Size']])
train_y = np.asanyarray(train[['CO2_Emissions']])


test_x = np.asanyarray(test[['Engine_Size']])
test_y = np.asanyarray(test[['CO2_Emissions']])

poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)

clf=linear_model.LinearRegression()
train_y_=clf.fit(train_x_poly,train_y)

plt.scatter(train.Engine_Size,train.CO2_Emissions,color='blue')
XX=np.arange(0.0,10.0,0.1)
yy=clf.intercept_[0]+clf.coef_[0][1]*XX+clf.coef_[0][2]*np.power(XX,2)
plt.plot(XX,yy,'-r')
plt.xlabel("Engine_Size")
plt.ylabel("CO2_Emissions")
plt.show()

test_x_poly = poly.fit_transform(test_x)
test_y_ = clf.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y,test_y_ ) )