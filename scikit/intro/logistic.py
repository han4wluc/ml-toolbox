#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lu
# @Date:   2016-01-26 21:13:47
# @Last Modified by:   lu
# @Last Modified time: 2016-01-27 10:01:44

import numpy as np
from sklearn import linear_model, datasets
import mltb

iris = datasets.load_iris()

XY = np.concatenate((iris.data, np.matrix(iris.target).T), axis=1)

XY = mltb.shuffle_rows(XY)

X = XY[:,0:-1];
y = XY[:,-1:];

X_train, X_test = mltb.sample(X)
y_train, y_test = mltb.sample(y)

logreg = linear_model.LogisticRegression(C=1e5)

logreg.fit(X_train, np.ravel(y_train))

prediction = logreg.predict(X_test)
correctCount = np.count_nonzero(prediction == np.ravel(y_test))

print 'accuracy: %f' % (correctCount / float(y_test.shape[0]))


