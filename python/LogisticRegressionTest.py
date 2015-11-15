import unittest
import LogisticRegression as LR
import numpy as np

class TestSigmoid(unittest.TestCase):
  def testOne(self):
    np.testing.assert_almost_equal(LR.sigmoid(1200000), 1)
    np.testing.assert_almost_equal(LR.sigmoid(-250), 0)
    np.testing.assert_almost_equal(LR.sigmoid(0), 0.5)

  def testTwo(self):
    z = np.array([4, 5, 6])
    expected = np.array([0.9820, 0.9933, 0.9975])
    np.testing.assert_almost_equal(LR.sigmoid(z), expected, decimal=4)
    
  def testThree(self):
    z = np.array([[8, 1, 6],
                  [3, 5, 7],
                  [4, 9, 2],])
    expected = np.array([[0.9997, 0.7311, 0.9975],
                         [0.9526, 0.9933, 0.9991],
                         [0.9820, 0.9999, 0.8808],])
    np.testing.assert_almost_equal(LR.sigmoid(z), expected, decimal=4)

# class TestPredict(unittest.TestCase):
#   def textOne(self):


class TestCostFunction(unittest.TestCase):
  def testOne(self):
    X = np.array([[8, 1, 6],
                  [3, 5, 7],
                  [4, 9, 2],
                  [8, 1, 6],
                  [3, 5, 7],
                  [4, 9, 2]])
    y = np.array([[1],
                  [0],
                  [1],
                  [0],
                  [1],
                  [0]])
    theta = np.array([[0],
                      [1],
                      [0]])
    expectedJ     = 2.6067
    expectedTheta = np.array([[1.7760],
                              [2.3988],
                              [1.9464]])
    [actualJ, actualTheta] = LR.costFunction(X, y, theta)
    np.testing.assert_almost_equal(actualJ, expectedJ, decimal=4)
    np.testing.assert_almost_equal(actualTheta, expectedTheta, decimal=4)
  
  def testTwo(self):
    X = np.array([[ 1,  1,  1,  1],
                  [ 1,  1,  1,  1],
                  [ 1,  1,  1,  1],
                  [ 1,  1,  1,  1],
                  [16,  2,  3, 13],
                  [ 5, 11, 10,  8],
                  [ 9,  7,  6, 12],
                  [ 4, 14, 15,  1]])
    y = np.array([[1],
                  [0],
                  [1],
                  [0],
                  [1],
                  [0],
                  [1],
                  [0]])
    theta = np.array([[0],
                      [1],
                      [0],
                      [1]])
    expectedJ     = 4.8135
    expectedTheta = np.array([[1.3154],
                              [3.3154],
                              [3.3154],
                              [1.3154]])
    [actualJ, actualTheta] = LR.costFunction(X, y, theta)
    np.testing.assert_almost_equal(actualJ, expectedJ, decimal=4)
    np.testing.assert_almost_equal(actualTheta, expectedTheta, decimal=4)


# TODO test failing
# class TestCostFunctionReg(unittest.TestCase):
#   def testOne(self):
#     X = np.array([[1, 8, 1, 6],
#                   [1, 3, 5, 7],
#                   [1, 4, 9, 2]]);
#     y = np.array([[1],
#                   [0],
#                   [1]])
#     theta = np.array([[-2],
#                       [-1],
#                       [ 1],
#                       [ 2]])
#     expectedJ_1     = 7.6832
#     expectedTheta_1 = np.array([[ 0.3172],
#                                 [-0.1277],
#                                 [ 2.6481],
#                                 [ 4.2379]])
#     [actualJ_1, actualTheta_1] = LR.costFunctionReg(X, y, theta, 3)
#     np.testing.assert_almost_equal(actualJ_1, expectedJ_1, decimal=4)
#     np.testing.assert_almost_equal(actualTheta_1, expectedTheta_1, decimal=4)

#     expectedJ_2     = 9.6832
#     expectedTheta_2 = np.array([[ 0.3172],
#                                 [-0.7944],
#                                 [ 3.3148],
#                                 [ 5.5712]])
#     [actualJ_2, actualTheta_2] = LR.costFunctionReg(X, y, theta, 5)
#     np.testing.assert_almost_equal(actualJ_2, expectedJ_2, decimal=4)
#     np.testing.assert_almost_equal(actualTheta_2, expectedTheta_2, decimal=4)


if __name__ == '__main__':
  all_tests = unittest.TestLoader().discover('tests', pattern='*.py')
  unittest.main()