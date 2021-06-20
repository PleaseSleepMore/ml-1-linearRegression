import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# dataset loading
data=pd.read_csv('./ex1data1.txt',header=None,names=['population','profit'])
data.head()

data.plot(kind='scatter',x='population',y='profit')
plt.show()

# theta.T 为矩阵专职
# gradient descent for single
def computeCost(x,y,theta):
    inner=np.power((x*theta.T-y),2)
    return np.sum(inner)/(2*len(x))


data.insert(0,'ones',1)

cols=data.shape[1]
X=data.iloc[:,0:cols-1]
Y=data.iloc[:,cols-1:cols]
# print(X.head())
# print(Y.head())

#matrix
X=np.matrix(X.values)
Y=np.matrix(Y.values)
theta=np.matrix([0,0])


print(X.shape)
print(Y.shape)
print(theta.shape)
print(computeCost(X,Y,theta))

#shape[i]第i维的长度
# batch gradient decent
def gradientDescent(x,y,theta,alpha,epoch):
    temp=np.matrix(np.zeros(theta.shape))
    parameters=int(theta.flatten().shape[1])
    cost=np.zeros(epoch)
    m=x.shape[0]

    for i in range(epoch):
        # simultaniously update params
        temp=theta-(alpha/m)*(x*theta.T-y).T*x
        theta=temp
        cost[i]=computeCost(x,y,theta)

    return theta,cost

#init
alpha=0.01
epoch=1000
# calculate
final_theta,cost=gradientDescent(X,Y,theta,alpha,epoch)

print(final_theta,cost)
print(computeCost(X,Y,final_theta))



# visualization

# h_theta(x)  vs  y
x=np.linspace(data.population.min(),data.population.max(),100)
# h_theta(x)
f=final_theta[0,0]+final_theta[0,1]*x

fig,ax=plt.subplots(figsize=(6,4))
ax.plot(x,f,'r',label='population')
ax.scatter(data['population'], data.profit, label='Traning Data')
ax.legend(loc=2)  # 2表示在左上角
ax.set_xlabel('population')
ax.set_ylabel('profit')
plt.show()


# cost function
fig,ax=plt.subplots(figsize=(8,4))
ax.plot(np.arange(epoch),cost,'r')
ax.set_xlabel('iterations')
ax.set_ylabel('cost')
plt.show()



# 多变量线性回归
data2=pd.read_csv('./ex1data2.txt',names=['size','bedrooms','price'])

# normalization
# data.mean 平均值  data.std() 标准差
data2=(data2-data2.mean())/data2.std()

# repeat process
data2.insert(0,'ones',1)
cols=data2.shape[1]
x2=data2.iloc[:,0:cols-1] #行，列 提取
y2=data2.iloc[:,cols-1:cols]

x2=np.matrix(x2.values)
y2=np.matrix(y2.values)
theta2=np.matrix(np.array([0,0,0]))

g2,cost2=gradientDescent(x2,y2,theta2,alpha,epoch)

print(g2)
print(computeCost(x2,y2,g2))

# visualize traing process
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(epoch), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()


