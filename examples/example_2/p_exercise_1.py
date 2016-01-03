#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lu
# @Date:   2015-11-22 16:02:02
# @Last Modified by:   lu
# @Last Modified time: 2015-11-22 16:14:50

import numpy as np
import sys
sys.path.append('../../python')
import LogisticRegression as LR

data1 = np.genfromtxt ('data1.txt', delimiter=",")
# np.loadtxt(open("test.csv","rb"),delimiter=",",skiprows=1)

m = data1.shape[0];

a = np.array([np.ones(m)]).T;
b = data1[:,0:2];
X = np.column_stack((a,b))
y = data1[:,2:3];

theta = np.array([[0],
                  [0],
                  [0]]) # initialize fitting parameters
iterations = 1000;
alpha = 0.001;

# print y.size

# for i in range(iterations):
#   predictions = X.dot(theta)
#   errors = (predictions - y) * X
#   print theta
#   theta = (theta.T - alpha * (1.0 / m) * errors.sum(axis=0)).T
# return   


[minTheta, JHist] = LR.gradientDescent(X, y, theta, alpha, iterations)
# print minTheta
