from sklearn import  linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# dataset loading
data=pd.read_csv('ex1data1.txt', header=None, names=['population', 'profit'])
data.insert(0,'ones',1)

cols=data.shape[1]
x=data.iloc[:,0:cols-1]
y=data.iloc[:,cols-1:cols]


model=linear_model.LinearRegression()
model.fit(x,y)
print(y)
plt.scatter(x.population,y.profit,color='red')
plt.plot(x.population,model.predict(x),color='blue')
plt.show()