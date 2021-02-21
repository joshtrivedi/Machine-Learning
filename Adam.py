#Creating the Dummy Data set and importing libraries
import math
import seaborn as sns
import numpy as np 
from scipy import stats
from matplotlib import pyplot as plt
x = np.random.normal(0,1,size=(100,1))
y = np.random.random(size=(100,1))
#finding the actual graph of linear regression and values of B0 and B1
print("Intercept is " ,stats.mstats.linregress(x,y).intercept)
print("Slope is ", stats.mstats.linregress(x,y).slope)
#Plotting 
plt.figure(figsize=(15,8))
sns.regplot(x,y)
plt.show()

#equation of straight lines
h = lambda theta_0, theta_1, x: theta_0 + np.dot(x,theta_1)

# the cost function (for the whole batch. for comparison later)
def J(x, y, theta_0, theta_1):
    m = len(x)
    returnValue = 0
    for i in range(m):
        returnValue += (h(theta_0, theta_1, x[i]) - y[i])**2
    returnValue = returnValue/(2*m)
    return returnValue

# finding the gradient per each training example
def grad_J(x, y, theta_0, theta_1):
    returnValue = np.array([0., 0.])
    returnValue[0] += (h(theta_0, theta_1, x) - y)
    returnValue[1] += (h(theta_0, theta_1, x) - y)*x
    return returnValue
#Adam Optimizer
class AdamOptimizer:
    def __init__(self, weights, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 0
        self.theta = weights
        
    def backward_pass(self, gradient):
        self.t = self.t + 1
        self.m = self.beta1*self.m + (1 - self.beta1)*gradient
        self.v = self.beta2*self.v + (1 - self.beta2)*(gradient**2)
        m_hat = self.m/(1 - self.beta1**self.t)
        v_hat = self.v/(1 - self.beta2**self.t)
        self.theta = self.theta - self.alpha*(m_hat/(np.sqrt(v_hat) - self.epsilon))
        return self.theta

#Checking for hyperparameters
epochs = 1500
print_interval = 100
m = len(x)
initial_theta = np.array([0., 0.]) # initial value of theta, before gradient descent
initial_cost = J(x, y, initial_theta[0], initial_theta[1])
theta = initial_theta
adam_optimizer = AdamOptimizer(theta, alpha=0.001)
adam_history = [] # to plot out path of descent
adam_history.append(dict({'theta': theta, 'cost': initial_cost}))
#training the model
for j in range(epochs):
    for i in range(m):
        gradients = grad_J(x[i], y[i], theta[0], theta[1])
        theta = adam_optimizer.backward_pass(gradients)
    
    if ((j+1)%print_interval == 0 or j==0):
        cost = J(x, y, theta[0], theta[1])
        print ('After {} epochs, Cost = {}, theta = {}'.format(j+1, cost, theta))
        adam_history.append(dict({'theta': theta, 'cost': cost}))
        
print ('\nFinal theta = {}'.format(theta))

#Plotting the final adam optimized linear regression plot
b = theta
yhat = b[0] + x.dot(b[1])
plt.figure(figsize=(15,8))
plt.scatter(x, y, color='red')
plt.plot(x, yhat, color='blue')
plt.show()
