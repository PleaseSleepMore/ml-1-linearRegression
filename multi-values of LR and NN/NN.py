import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat               # 将数据从MATLAB格式加载到python
from scipy.optimize import minimize
from sklearn.metrics import classification_report

# data loading
weight=loadmat("./ex3weights.mat")
theta1,theta2=weight["Theta1"],weight['Theta2']

data=loadmat("./ex3data1.mat")
X2 = np.matrix(np.insert(data['X'], 0, values=np.ones(data['X'].shape[0]), axis=1))
y2 = np.matrix(data['y'])

def Sigmoid(z):
    return 1/(1+np.exp(-z))

# travel forward
# layer1
a1=X2
z2=a1*theta1.T
a2=Sigmoid(z2)
# layer2
a2=np.insert(a2,0,values=np.ones(a2.shape[0]),axis=1)
z3=a2*theta2.T
a3=Sigmoid(z3)

y_pred2=np.argmax(a3,axis=1)+1
print(classification_report(y2,y_pred2))

