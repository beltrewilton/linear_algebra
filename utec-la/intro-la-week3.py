import numpy as np
from scipy.sparse import csr_matrix
from numpy import tensordot
from scipy.linalg import lu
from numpy.linalg import qr
from numpy.linalg import eig
from numpy.linalg import inv
from numpy import diag
from numpy.linalg import cholesky
import math
from numpy.linalg import norm
from numpy import array

# lecture week 3.1 extensions

A = np.random.randint(0, 5, (15, 23))
S = csr_matrix(A)


# print(A)


def sparse_it(matx):
    sparce_matx = []
    for i, outside in enumerate(matx):
        tmp = []
        for j, inside in enumerate(outside):
            if inside != 0:
                tmp.append((i, j, inside))
        if len(tmp) > 0:
            sparce_matx.append(tmp)
    return sparce_matx


def sparse_recursive(matx):
    result = []
    if type(matx) is not None:
        if len(matx) < 1:
            return 1
    for e in matx:
        return sparse_recursive(result.append(e))



# M = np.random.randint(0,2, (15,23))
M = np.array([[1, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0]])
# print(M)
# print(sparse_it(M))

print(sparse_recursive(M))


# lecture week 3.2 extensions
A = np.array([[1, 0, 0, 1, 0, 0],
       [0, 0, 2, 0, 0, 1],
       [0, 0, 0, 2, 0, 0]])
B = 2 * A
# print(tensordot(A, B, axes=0))

K = np.array([1,2,3])
L = np.array([4,7,9])
J = tensordot(K, L, axes=0)
# print(L)

# 3x3 tensor basics diffusion operations.
def tensor_basics(tensor_1, tensor_2, op):
    tensor_result = []
    for i, toplevel in enumerate(tensor_1):
        l1 = []
        for j, mediumlevel in enumerate(toplevel):
            l2 = []
            for k, value in enumerate(mediumlevel):
                if "+" == op:
                    l2.append(value + tensor_2[i][j][k])
                elif "-" == op:
                    l2.append(value - tensor_2[i][j][k])
                elif "*" == op:
                    l2.append(value * tensor_2[i][j][k])
                elif "/" == op:
                    l2.append(value / tensor_2[i][j][k])
                else:
                    raise('Debe indicar un tipo de operacion +, -, *, /')
            l1.append(l2)

        tensor_result.append(l1)
    return np.array(tensor_result)

T = np.array([
        [[1,2,3], [4,5,6], [7,8,9]],
        [[11,12,13], [14,15,16], [17,18,19]],
        [[21,22,23], [24,25,26], [27,28,29]]
    ])

# print(tensor_basics(T, T, '/'))
# print('------------------------------')
# print(T/T)


# lecture week 3.3 extensions

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
D = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])

# print(A)
# # print(lu(A))
# P, L, U = lu(A)
# print(P@L@U)

Q, R = qr(D, 'complete')
# print(D)
# print(Q@R)
#
#
# H = cholesky(D)
# print(H)
# print(H@H.T)
#


# lecture week 3.4 extensions
X = np.array([[1,2,3], [1,5,9], [9,3,7]])
values, vectors = eig(X)
# print(values, '\n\n', vectors)
#confirm:
Z = X @ vectors[:, 0]
K = vectors[:, 0] * values[0]
# print(Z, '\n\n', K)
#reconstruir
Q = vectors
R = inv(Q)
L = diag(values)
# print(Q @ L @ R)
"""
    Implemente la operación de descomposición propia desde cero para matrices definidas
    como listas de listas.
"""
