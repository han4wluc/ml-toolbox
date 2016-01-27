#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lu
# @Date:   2016-01-25 21:17:41
# @Last Modified by:   lu
# @Last Modified time: 2016-01-26 09:12:11

import numpy as np
import mltb
from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()
diabetes = datasets.load_diabetes()



# mltb.summary(iris.data);
# mltb.column(iris.data);

shape = iris.data.shape;

m = shape[0];
n = shape[1];

X = iris.data[:,0:-1];
y = iris.data[:,-1:];

# print X;
# print y;

print X.shape
print np.ones((m,1));

# X = np.concatenate((np.ones((m,1)), X), axis=1)
# print X

# print X[:-2]; all rows except last tow
# print X[-2:]; last two rows

# print X;
# from sklearn import linear_model;
# clf = linear_model.LinearRegression();
# clf.fit (X, y);
# print clf.coef_
# LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
# clf.coef_
# array([ 0.5,  0.5])

# mltb.export_csv(iris.data, './data/iris.csv');
# mltb.export_csv(digits.data, './data/digits.csv');
# mltb.export_csv(diabetes.data, './data/diabetes.csv');

