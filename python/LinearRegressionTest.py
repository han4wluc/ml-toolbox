import unittest
import LinearRegression as LR
import numpy as np

class TestFeatureNormalize(unittest.TestCase):
  def testOne(self):
    X = np.array([[3],
                  [4],
                  [5]])
    expected = np.array([[-1.],
                         [ 0.],
                         [ 1.]])
    np.testing.assert_almost_equal(LR.featureNormalize(X), expected)

class TestComputeCost(unittest.TestCase):
  def testOne(self):
    X = np.array([[1, 2],
                  [1, 3],
                  [1, 4],
                  [1, 5]])
    y = np.array([[ 7.],
                  [ 6.], 
                  [ 5.], 
                  [ 4.]]);
    theta = np.array([[0.1],
                      [0.2]])
    expected = 11.9450
    np.testing.assert_almost_equal(LR.computeCost(X, y, theta), expected)

  def testTwo(self):
    X = np.array([[1, 2, 3],
                  [1, 3, 4],
                  [1, 4, 5],
                  [1, 5, 6]])
    y = np.array([[ 7.],
                  [ 6.],
                  [ 5.],
                  [ 4.]])
    theta = np.array([[0.1],
                      [0.2],
                      [0.3]])
    expected = 7.0175;
    np.testing.assert_almost_equal(LR.computeCost(X, y, theta), expected)

class TestGradientDescent(unittest.TestCase):
  def testOne(self):
    X = np.array([[1, 5],
                  [1, 2],
                  [1, 4],
                  [1, 5]])
    y = np.array([[1],
                  [6],
                  [4],
                  [2]])
    theta = np.array([[0],
                      [0]])
    alpha = 0.01;
    numOfIter = 1000;
    expectedTheta = np.array([[ 5.2148],
                              [-0.5733]])
    # expectedJHist[0] = 0.85426; 
    [actualTheta, actualJHist] = LR.gradientDescent(X, y, theta, alpha, numOfIter)
    np.testing.assert_almost_equal(actualTheta, expectedTheta, decimal=4);

# class TestNormalEquation(unittest.TestCase):
#   def testOne(self):
#     X = np.array([[1, 5],
#                   [1, 2],
#                   [1, 4],
#                   [1, 5]])
#     y = np.array([[1],
#                   [6],
#                   [4],
#                   [2]])
#     expectedTheta = np.array([[ 5.2148],
#                               [-0.5733]])
#     # expectedJHist[0] = 0.85426; 
#     actualTheta = LR.normalEquation(X, y)
#     np.testing.assert_almost_equal(actualTheta, expectedTheta, decimal=4)

if __name__ == '__main__':
    unittest.main()