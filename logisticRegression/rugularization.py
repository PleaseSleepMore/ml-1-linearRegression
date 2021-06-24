import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_init=pd.read_csv("./ex2data2.txt",header=None,names=['test1','test2','accepted']);


positive=data_init[data_init['accepted'].isin([1])]
negative=data_init[data_init['accepted'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['test1'], positive['test2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative['test1'], negative['test2'], s=50, c='r', marker='x', label='Rejected')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
plt.show()

# 此数据集不能使用线性分割，所以不适合直接使用逻辑回归

# 特征映射
# 如果样本量多，逻辑回归问题很复杂
# 而原始特征只有x1,x2可以用多项式创建更多的特征x1、x2、x1x2、x1^2、x2^2、... X1^nX2^n。
# 因为更多的特征进行逻辑回归时，得到的分割线可以是任意高阶函数的形状。
degree=6
data2=data_init
ori1=x1=data2['test1']
ori2=x2=data2['test2']
print("original data2:\n")
print(data2.head())

data2.insert(3,'ones',1)

for i in range(1,degree+1):
    for j in range (0,i+1):
        data2['F'+str(i-j)+str(j)]=np.power(x1,i-j)*np.power(x2,j)

data2.drop('test1',axis=1,inplace=True)
data2.drop('test2', axis=1, inplace=True)

print("data2 after feature mapping:\n")
print(data2.head())

# sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))

# cost function and gredient
def cosrReg(theta,x,y,learningRate):
    theta=np.matrix(theta)
    x=np.matrix(x)
    y=np.matrix(y)
    first=np.multiply(-y,np.log(sigmoid(x*theta.T)))
    second=np.multiply(1-y,np.log(1-sigmoid(x*theta.T)))
    reg=learningRate/(2*len(x))*np.sum(np.power(theta[:,1:theta.shape[1]],2))
    return np.sum(first-second)/len(x)+reg

# 因为用使用计算库来求，所以只需要求梯度即可（J关于每个参数的偏导 0单独分类）
def gradientReg(theta,x,y,learningRate):
    theta=np.matrix(theta)
    x=np.matrix(x)
    y=np.matrix(y)

    parameters=int(theta.ravel().shape[1])
    grad=np.zeros(parameters)

    error=sigmoid(x*theta.T)-y

    for i in range(parameters):
        term=np.multiply(error,x[:,i])

        if(i==0):
            grad[i]=np.sum(term)/len(x)
        else:
            grad[i]=(np.sum(term)/len(x))+learningRate/len(x)*theta[:,i]

    return grad

cols=data2.shape[1]
x2=data2.iloc[:,1:cols]
y2=data2.iloc[:,0:1]
theta2=np.zeros(cols-1)

x2=np.array(x2.values)
y2=np.array(y2.values)

learningrate=1

print(cosrReg(theta2,x2,y2,learningrate))

import scipy.optimize as opt
result=opt.fmin_tnc(func=cosrReg,x0=theta2,fprime=gradientReg,args=(x2,y2,learningrate))
print(result)


# 通过预测函数 查看准确度
# predict and revalue
def hfunc2(theta,x1,x2):
    temp=theta[0][0]
    place=0
    for i in range(1,degree+1):
        for j in range(0,i+1):
            temp+=np.power(x1, i-j) * np.power(x2, j) * theta[0][place+1]
            place+=1
    return temp
print(sigmoid(hfunc2(result,ori1,ori2)))


# correct rate
# correct answer
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

theta_min = np.matrix(result[0])
predictions = predict(theta_min, x2)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y2)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))
