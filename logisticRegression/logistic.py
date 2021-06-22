import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("./ex2data1.txt",header=None,names=['Exam1','Exam2','Admitted']);
print(data.head());

# output scatter with 2 colors representing 2 admitted status
positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Exam1'], positive['Exam2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam1'], negative['Exam2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()

# sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))

# cost function
def cost(theta,x,y):
    theta=np.matrix(theta)
    x=np.matrix(x)
    y=np.matrix(y)
    first=np.multiply(-y,np.log(sigmoid(x*theta.T)))
    second=np.multiply((1-y),np.log(1-sigmoid(x*theta.T)))
    return np.sum(first-second)/len(x)

# init
data.insert(0,'ones',1)

cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]
theta = np.zeros(3)

x=np.array(X.values)
y=np.array(y.values)

print(x.shape,y.shape,theta.shape)
print(cost(theta,x,y))

# gradient calculate not performing descend process
def gradient(theta,x,y):
    theta=np.matrix(theta)
    x=np.matrix(x)
    y=np.matrix(y)

    parameters=int(theta.ravel().shape[1])
    grad=np.zeros(parameters)

    error=sigmoid(x*theta.T)-y
    for i in range(parameters):
      term=np.multiply(error,x[:,i])
      grad[i]=np.sum(term)/len(x)

    return grad

# use optimize to execute gredient descend
import scipy.optimize as opt
result=opt.fmin_tnc(func=cost,x0=theta,fprime=gradient,args=(x,y))
# fmin_tnc return:x数组(优化问题的目标值)， nfeval,rc
print(result[0].shape,x.shape,y.shape)
print(x)
print(result)
cost(result[0],x,y)

# visualization
# y axs is exam2 and x axs is exam1, so it need transfering.
plotting_x1 = np.linspace(30, 100, 100)
plotting_h1 = ( - result[0][0] - result[0][1] * plotting_x1) / result[0][2]

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(plotting_x1, plotting_h1, 'y', label='Prediction')
ax.scatter(positive['Exam1'], positive['Exam2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam1'], negative['Exam2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()

# predict and revalue
def hfunc1(theta,x):
    return sigmoid(np.dot(theta.T,x))

print(hfunc1(result[0],[1,45,85]))
# correct rate
def predict(theta,x,y):
   counter=0
   for i in range(len(x)):
       if((hfunc1(theta,x[i])>0.5 and y[i]==1)or(hfunc1(theta,x[i])<0.5 and y[i]==0)):
           counter=counter+1
   return counter/len(x)

print(predict(result[0],x,y))



