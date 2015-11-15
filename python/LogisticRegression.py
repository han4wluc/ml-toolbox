import numpy as np

def sigmoid(z):
  '''
  Compute sigmoid function
  '''
  return 1 / (1 + np.e ** -z)

def costFunction(X, y, theta):
  '''
  Computes Logistical cost funcion, return Cost, Gradient
  '''

  m = X.shape[0]
  h = sigmoid(X.dot(theta))
  J = 1./m * (-y.flatten() .dot( np.log(h)) - ( 1 - y.flatten() ) .dot( np.log(1-h)))
  g = 1./m * X.T.dot(h - y)

  return J, g

  # grad = 1./m * X' * (sigmoid(X * theta) - y);
  # sigmoid(X * theta)
  # 1./m * ( -y' * log( sigmoid(X * theta) ) - ( 1 - y' ) * log ( 1 - sigmoid( X * theta))


def costFunctionReg(X, y, theta, l):
  '''
  Computes Regularized logistical regeression
  '''
  m = X.shape[0]
  h = sigmoid(X.dot(theta))
  unregularizedJ = 1./m * (-y.flatten() .dot( np.log(h)) - ( 1 - y.flatten() ) .dot( np.log(1-h)));

  g = 1./m * X.T.dot(h - y)
  gr = l / m * theta
  gFinal = g + gr;
  gFinal[0] = g[0];
  r = (l /  ( 2. * m)) * (theta[:m] ** 2.).sum()
  return unregularizedJ + r, gFinal

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
    J_history[i, 0] = costFunction(X, y, theta)

  return theta, J_history

def gradient_descent_reg(X, y, theta, alpha, num_iters, l):
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
    J_history[i, 0] = costFunctionReg(X, y, theta, l)

  return theta, J_history

# def predict():
  # if sigmoid(theta'*x) >= 0.5, predict 1


