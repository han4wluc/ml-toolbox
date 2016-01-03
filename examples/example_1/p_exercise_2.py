# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # @Author: lu
# # @Date:   2015-11-22 15:49:05
# # @Last Modified by:   lu
# # @Last Modified time: 2015-11-22 15:53:52

# # m = size(data2, 1);
# # X = [ones(m, 1), data2(:,1:2)];

# # % X = featureNormalize(X)
# # % X_reg = featureNormalize(X)
# # theta = zeros(3, 1); % initialize fitting parameters
# # iterations = 1500;
# # alpha = 0.01;
# # y = data2(:,3);


# import numpy as np
# import sys
# sys.path.append('../../python')
# import LinearRegression as LR

# data2 = np.genfromtxt ('data2.txt', delimiter=",")
# # np.loadtxt(open("test.csv","rb"),delimiter=",",skiprows=1)

# m = data2.shape[0];

# a = np.array([np.ones(m)]).T;
# b = data2[:,0:2];
# X = np.column_stack((a,b))
# y = data[:,2:3];


# theta = np.array([[0],
#                   [0]]) # initialize fitting parameters
# iterations = 5000;
# alpha = 0.02;

# [minTheta, jHist] = LR.gradientDescentMulti(X, y, theta, alpha, iterations)


