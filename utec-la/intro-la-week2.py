import numpy as np
import math
from numpy.linalg import norm
from numpy import array

# lecture week 2.1 extensions
m = np.random.randint(1, 10, size=(2, 5))
# print(m)
# print(norm(m, 1))
# print(norm(m, 2))
# print(norm(m, np.inf))


# lecture week 2.2 extensions

def norm_l1(matx):
    acumm = 0
    for i in matx:
        for x in i:
            acumm += x
    return acumm

def norm_l2(matx):
    acumm = 0
    for i in matx:
        for x in i:
            acumm += x**2
    return math.sqrt(acumm)

def norm_max(matx):
    max = 0
    for i in matx:
        for x in i:
            if x > max:
                max = x
    return max

v = np.array([[5,6,7], [1,2,3]])
# print(norm_l2(v))
# print(norm(v, 2))
# print(norm_max(v))



# lecture week 2.3 extensions
"""
  > https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c
"""


# lecture week 2.4 extensions
B = np.random.randint(0,50, size=(3,5))
A = np.random.randint(0,50, size=(5,2))
v = np.array([4, 9])
# print(B.dot(A))
# print(A.dot(v))
# print(A-B)


# lecture week 2.5 extensions

def matx_basics(matx1, matx2, op):
    matx_result = []
    for idx1, val1 in enumerate(matx1):
        tmp = []
        for idx2, val2 in enumerate(val1):
            if "+" == op:
                tmp.append(val2 + matx2[idx1][idx2])
            elif "-" == op:
                tmp.append(val2 - matx2[idx1][idx2])
            elif "*" == op:
                tmp.append(val2 * matx2[idx1][idx2])
            elif "/" == op:
                tmp.append(val2 / matx2[idx1][idx2])
            else:
                raise('Debe indicar un tipo de operacion +, -, *, /')

        matx_result.append(tmp)
    return np.array(matx_result)

Z = np.array([[2,3,4], [10,20,30]])
M = np.array([[1,2,3], [5,6,7]])

# print(matx_basics(Z, M, "-"))

R = np.array([[3,4,5], [1,2,4], [6,7,8], [6,2,5], [1,3,2]])
T = np.array([[4,7,2,5], [0,3,1,2], [9,2,5,1]])
# print(R.shape)
# print('------')
# print(T.shape)
# print('------')
# print((R@T).shape)
# print('------')

def matx_dot_product(matx1, matx2):
    if matx1.shape[1] != matx2.shape[0]:
        raise('[n x m] debe ser igual a [m x k]')

    matx_result = []
    for idx1 in range(matx1.shape[0]):
        tmp = []
        # print('-------------')
        for idx2 in range(matx2.shape[1]):
            # print('{}x{}={}'.format(matx1[idx1], matx2[:,idx2], sum(matx1[idx1] * matx2[:,idx2])))
            # se puede aplica otro [for] para multiplicar 'manual' cada indice., pero no es necesario si ya puedo multiplicar array x array.
            tmp.append(sum(matx1[idx1] * matx2[:,idx2]))

        matx_result.append(tmp)
    return np.array(matx_result)

# R = np.random.randint(10,20, size=(4,6))
# T = np.random.randint(10,20, size=(6,8))
#
# print(matx_dot_product(R, T))
# print('')
# print(R@T)


# lecture week 2.6 extensions
# simetrica
Ms = np.array([[9,8,7,6], [8,9,8,7], [7,8,9,8], [6,7,8,9]])
# print(Ms.T == Ms)
# triangular
Mt = np.array([[9,8,7,6], [0,9,8,7], [0,0,9,8], [0,0,0,9]])
# lower
# print(Mt)
# upper
# print(Mt.T)
# print(np.tril(Ms))
# print(np.triu(Ms))
#diagonal
# print(np.diag(Ms))
# print(np.diag(np.diag(Ms)))
#identity
# print(np.identity(7))
#orthogonal
V = np.linalg.inv(Ms)
# print(V)
# print(Ms.T)
# print(Ms@Ms.T)


def simetrica(n):
    return np.random.randint(1,10, size=(n,n))


def triangular(n):
    m = simetrica(n)
    for idx in range(1, m.shape[0]):
        for idx2 in range(0, idx):
            m[idx, idx2] = 0

    return m


def diagonal(n):
    m = simetrica(n)
    tmp = []
    for idx in range(0, m.shape[0]):
        for idx2 in range(idx, idx+1):
            tmp.append(m[idx, idx2])

    return np.array(tmp)

def diagonal_mtx(mtx):
    size = mtx.shape[0]
    zeros = np.zeros((size, size))
    for idx in range(0, size):
        for idx2 in range(idx, idx+1):
            zeros[idx, idx2] = mtx[idx]

    return zeros


def identity(n):
    zeros = np.zeros((n, n))
    for idx in range(0, n):
        for idx2 in range(idx, idx+1):
            zeros[idx, idx2] = 1

    return zeros

# print(simetrica(14))
# print(triangular(9))
# d = diagonal(4)
# print(d)
# print(diagonal_mtx(d))
print(identity(8))
