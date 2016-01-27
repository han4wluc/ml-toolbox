#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lu
# @Date:   2016-01-24 23:44:12
# @Last Modified by:   lu
# @Last Modified time: 2016-01-27 09:58:57

import numpy as np
import sklearn as sl

def import_csv(filename):
  return np.loadtxt(fname = filename, delimiter = ',');

def export_csv(array, filename):
  # np.asarray([ [1,2,3], [4,5,6], [7,8,9] ])
  # np.savetxt(filename, array, delimiter=",", fmt='%1.1f')
  np.savetxt(filename, array, delimiter=",", newline='\r\n', fmt='%.5f')


# x = numpy.array([1,0,2,0,3,0,4,5,6,7,8])
# >>> numpy.where(x == 0)[0]
# array([1, 3, 5])
def summary(array):
  # print array.size;
  shape = array.shape;
  print "rows: %d columns: %d" % (shape[0],shape[1]); 
  print "max: %.2f" % np.amax(array)
  print "min: %.2f" % np.amin(array)
  print "median: %.2f" % np.median(array)
  print "average: %.2f" % np.average(array)
  print "standard deviation: %.2f" % np.std(array)
  print "variance: %.2f" % np.var(array)
  # print np.where(array > 4)[0].size

# >>> A = np.array([[1,2,3,4],[5,6,7,8]])

# >>> A
# array([[1, 2, 3, 4],
#     [5, 6, 7, 8]])

# >>> A[:,2] # returns the third columm
# array([3, 7])

def transpose(a):
  print numpy.transpose(a);
  return numpy.transpose(a);

def column(a):
  print a[:,0:1]; #get first column
  # coolumn = a[0:5,:] get first five rows 
  # print coolumn;
  # print a[0:20,:];
  # print X[:-2]; all rows except last tow
  # print X[-2:]; last two rows

def row(a):
  a[:,0:-1]; # all columns expect last one
  a[:,-1:];  # last column

def vector_to_matrix(a1, a2):
  return np.matrix([a1, a2]).T

def concatenate_matrix(a1,a2):
  return np.concatenate((a1, a2), axis=1)

def shuffle_rows(a):
  return sl.utils.shuffle(a)

def sample(a, size=0.4):
  m_test = round(a.shape[0] * size)
  return (a[:-m_test], a[-m_test:])
