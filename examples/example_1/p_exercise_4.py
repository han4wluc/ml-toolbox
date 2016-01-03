# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # @Author: lu
# # @Date:   2015-11-22 15:54:14
# # @Last Modified by:   lu
# # @Last Modified time: 2015-11-22 15:56:10


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
# y = data2[:,2:3];

# theta = LR.normalEquation(X, y)
# print theta


