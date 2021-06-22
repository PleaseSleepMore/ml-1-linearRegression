# ml-1-linearRegression
reference：https://blog.csdn.net/Cowry5/article/details/80174130

###compute cost
```python
def computeCost(x,y,theta):
    inner=np.power((x*theta.T-y),2)
    return np.sum(inner)/(2*len(x))
```

###batch gradient decent
```python
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

```


###sklearn
```python
from sklearn import  linear_model

model=linear_model.LinearRegression()
model.fit(x,y)
```