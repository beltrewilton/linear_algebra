import numpy as np
from math import e
import math
import pandas as pd
import timeit

# lecture 1 extensions
#
# euler = np.linspace(0, e, 10)
# print(euler)
#
# _eye_ = np.eye(3,5)
# print(_eye_)
#
# ident = np.identity(4)
# print(ident)
#
# a = np.array([1, 5, 8, 0])
# b = np.array([0.45, .5, .89, 1])
# print(np.concatenate((a,b)))
#
# print(np.dstack((b,a)))
# print(np.column_stack((a,b)))

# lecture 2 extensions

data = np.array([
         [ 2,   2,   2,    941,   1,     94905],
         [ 2,   3,   2,   1146,   0,     98937],
         [ 2,   3,   2,    909,   0,    100309],
         [ 3,   3,   2,   1289,   0,    106250],
         [ 3,   3,   1,   1020,   0,    107502],
         [ 2,   2,   2,   1022,   0,    108750],
         [ 2,   2,   2,   1134,   1,    110700],
         [ 2,   2,   1,    844,   0,    113263],
         [ 2,   2,   1,    795,   1,    116250],
         [ 2,   2,   1,    588,   0,    120000],
         [ 2,   3,   2,   1356,   0,    121630],
         [ 2,   3,   2,   1118,   0,    122000],
         [ 3,   4,   2,   1329,   0,    122682],
         [ 2,   4,   2,   1240,   0,    123000]])

X = data[:, :-1]
y = data[:, -1]
# print(X)
# print(y)

tpercent = math.floor(len(data) * 0.8)
train, test = data[:tpercent, :], data[tpercent:, :]
# print(train)
# print('{}'.format('-'*100))
# print(test)

cars = pd.read_csv('imports-85.data', nrows=20)
# cars = cars.truncate(1, 20)
c_percent = math.floor(len(cars) * 0.8)
cars_train, cars_test = cars[:c_percent], cars[c_percent:]
# print(cars_train)
# print('{}'.format('-'*100))
# print(cars_test)

# lecture 3 extensions

employees = np.array([[1, 'Wilton Beltre', 30000.0, 0.023],
                      [2, 'Martin Sorondo', 35000.0, 0.027],
                      [3, 'Ana Maria', 42000.0, 0.031]])
employees[:, -2:-1] = employees[:, -2:-1].astype(np.float) * 1.045

def sum_oned_array(a, s):
    return np.array([(i+s) for i in a])

def sum_twod_array(a, s):
    return np.array([[(m+s) for m in i] for i in a])

# print(sum_oned_array(np.array([1,2,3]), 4))
# print(sum_twod_array(np.array([[1,2,3], [2,4,6]]), 4))




n = 10000000
sum = 100
# large = np.array([np.random.random(n), np.random.random(n)])
# start = timeit.default_timer()
# r = sum_twod_array(large, sum)
# stop = timeit.default_timer()
# print('my function time spent: {} secs.'.format(stop - start))
#
# start = timeit.default_timer()
# r2 = large + sum
# stop = timeit.default_timer()
# print('native numpy broadcasting time spent: {} secs.'.format(stop - start))

# lecture 4 extensions
# a = np.array([1, 2, 3])
# b = np.array([[1, 2, 3], [5, 6, 7]])
# print(b.dot(a))
# print(a/b)
# print(a*b)

