import numpy as np
from matplotlib import pyplot as plt

#creating our data
X = np.random.rand(10,1)
y = np.random.rand(10,1)

#Computing coefficient
b = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

#Predicting our Hypothesis
yhat = X.dot(b)
#Plotting our results
plt.scatter(X, y, color='red')
plt.plot(X, yhat, color='blue')
plt.show()