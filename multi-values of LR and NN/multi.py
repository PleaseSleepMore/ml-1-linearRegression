import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat               # 将数据从MATLAB格式加载到python
from scipy.optimize import minimize
from sklearn.metrics import classification_report

data=loadmat("./ex3data1.mat")
X=data["X"]
y=data["y"]
# print(data['X'].shape)
# print(data['y'].shape)
X=np.insert(X,0,values=np.ones(X.shape[0]),axis=1)
n=X.shape[1]
m=X.shape[0]

def Sigmoid(z):
    return 1/(1+np.exp(-z))

def CostFunction(theta,X,y,learningRate):
    theta=np.matrix(theta)
    x=np.matrix(X)
    y=np.matrix(y)
    first=np.multiply(-y,np.log(Sigmoid(X*theta.T)))
    second=np.multiply((1-y),np.log(1-Sigmoid(X*theta.T)))
    reg=learningRate/(2*len(X))*np.sum(np.power(theta[:,1:theta.shape[1]],2))
    return np.sum(first-second)/len(X)+reg

def GradientDescent(theta,X,y,learningRate):
    theta=np.matrix(theta)
    X=np.matrix(X)

    parameters=int(theta.ravel().shape[1])
    error=Sigmoid(X*theta.T)-y

    grad=((X.T*error)/len(X)).T + learningRate/len(X)*theta

    grad[0,0]=np.sum(np.multiply(error,X[:,0]))/len(X)

    return np.array(grad).ravel()

def one_vs_all(X,y,num_labels,learningRate):
    theta_all=np.zeros((num_labels,n))

    for i in range(1,num_labels+1):
        theta_i=np.zeros(n)
        y_i=np.array([1 if label == i else 0 for label in y])
        y_i=np.reshape(y_i,(m,1))
        theta_i=theta_i.reshape(theta_i.shape[0],1)

        fmin=minimize(CostFunction,theta_i,args=(X,y_i,learningRate),method='TNC',jac=GradientDescent)
        theta_all[i-1:]=fmin.x

    return theta_all

theta_all=one_vs_all(X,y,10,1)

def predict_all(X,theta_all):
    X=np.matrix(X)
    theta_all=np.matrix(theta_all)

    h=Sigmoid(X*theta_all.T)
    h_argmax=np.argmax(h,axis=1)

    return h_argmax+1

y_pred = predict_all(X, theta_all)
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
accuracy = np.sum(correct) / 5000.0
print('accuracy = {0}%'.format(accuracy * 100))
print(classification_report(data['y'],y_pred))