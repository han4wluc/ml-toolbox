#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lu
# @Date:   2015-11-22 13:18:39
# @Last Modified by:   lu
# @Last Modified time: 2015-11-22 15:48:47

import numpy as np
# import '../../python/LinearRegression.py' as LR

import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('../../python')

import LinearRegression as LR


data = np.genfromtxt ('data.txt', delimiter=",")
# np.loadtxt(open("test.csv","rb"),delimiter=",",skiprows=1)

# data[0:0]

# m = size(data, 1);
# X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
# theta = zeros(2, 1); % initialize fitting parameters
# iterations = 5000;
# alpha = 0.02;
# y = data(:,2);

m = data.shape[0];

a = np.array([np.ones(m)]).T;
b = data[:,0:1];
X = np.column_stack((a,b))
y = data[:,1:2];


theta = np.array([[0],
                  [0]]) # initialize fitting parameters
iterations = 5000;
alpha = 0.02;

[minTheta, jHist] = LR.gradientDescent(X, y, theta, alpha, iterations)
# print minTheta  #-3.89578082    1.19303364
# print (minTheta[0] + 3.5 * minTheta[1])[0]  #0.27983691
# print (minTheta[0] +   7 * minTheta[1])[0] #4.45545465
# predictedY = theta[0] + (X[:,1:2] * minTheta[1])

