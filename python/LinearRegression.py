import numpy as np

# @param type:vector 
# 
# 
def featureNormalize(X):
  '''
  Returns a normalized version of X where
  the mean value of each feature is 0 and the standard deviation
  is 1. This is often a good preprocessing step to do when
  working with learning algorithms.
  '''
  
  mean = np.mean(X, axis=0)
  std = np.std(X, axis=0, ddof=1)
  return (X - mean)/std

# @param type:matrix 
# @param type:matrix
# @param type:number
def computeCost(X, y, theta):
  '''
  Compute cost for linear regression
  '''

  m = y.size
  predictions = X.dot(theta)
  sqErrors = (predictions - y) ** 2
  J = (1.0 / (2 * m)) * sqErrors.sum()
  return J


def gradientDescent(X, y, theta, alpha, num_iters):
  '''
  Performs gradient descent to learn theta 
  by taking num_items gradient steps with 
  learning rate alpha
  '''

  m = y.size
  J_history = np.zeros(shape=(num_iters, 1))

  for i in range(num_iters):
    predictions = X.dot(theta)
    errors = (predictions - y) * X 
    theta = (theta.T - alpha * (1.0 / m) * errors.sum(axis=0)).T
    J_history[i, 0] = computeCost(X, y, theta)

  return theta, J_history


def normalEquation(X, y):
  '''
  Computes the closed-form solution to linear regression
  '''
  return np.linalg.inv((X.T.dot(X))).dot(X.T).dot(y.T)









