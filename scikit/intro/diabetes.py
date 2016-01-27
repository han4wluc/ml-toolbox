#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lu
# @Date:   2016-01-26 09:07:00
# @Last Modified by:   lu
# @Last Modified time: 2016-01-26 10:05:15
import numpy as np
from sklearn import datasets, linear_model

diabetes = datasets.load_diabetes()

diabetes_X = diabetes.data[:, np.newaxis, 2] #diabetes.data[:,2:3]

diabetes_X_train = diabetes_X[:-20]
diabetes_X_test  = diabetes_X[-20:]

diabetes_y_train = diabetes.target[:-20]
diabetes_y_test  = diabetes.target[-20:]

regr = linear_model.LinearRegression()

regr.fit(diabetes_X_train, diabetes_y_train)

print 'Coefficients: \n', regr.coef_

print 'Residual sum of squared: %.2f' \
       % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2)

print 'Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test)

diabetes_X_predicted = np.round(regr.predict(diabetes_X_test))
result = np.matrix([diabetes_X_predicted, diabetes_y_test]).T
# print result[:,0:1];
# print result[:,1:2];
diff = result[:,0:1] - result[:,1:2]

print np.concatenate((result, diff), axis=1);

# print diabetes_y_test
# print [diabetes_X_predicted.T, diabetes_y_test.T]
# print np.concatenate((diabetes_X_predicted, diabetes_y_test), axis=1)

# print np.round(diabetes_X_predicted - diabetes_y_test)
