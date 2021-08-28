import numpy as np
from numpy.linalg import cholesky
from numpy.linalg import eig


"""
    1) implemente una función que reciba un valor de densidad (float) y el número de elementos
    de una matriz (valor entero) y calcule la cantidad de elementos no cero que debería tener la
    matriz para esa densidad (debe retornar un entero).
    >> density: sum of non-zero elements.
"""


def count_non_zero(density, n):
    return density * n


A = np.array([[1, 0, 0, 1, 0, 0], [0, 0, 2, 0, 0, 1], [0, 0, 0, 2, 0, 7]])
density = np.count_nonzero(A) / A.size
print('(1) cantidad de elementos no cero: ', count_non_zero(density, A.size))


"""
    2) implemente una función que reciba por parámetro dos tensores A y B e imprima por
    pantalla:
    • el producto Hadamard entre ellos y sus dimensiones
    • el producto Tensor entre ellos y sus dimensiones
"""


def tensors_product(tensor_1, tensor_2, alg='tensor_product'):
    tensor_result = []
    if alg == 'tensor_product':
        tensor_result = np.tensordot(tensor_1, tensor_2, axes=0)
    # Hadamard [3x3] product hand implemented for fun.
    elif alg == 'Hadamard':
        for i, toplevel in enumerate(tensor_1):
            l1 = []
            for j, mediumlevel in enumerate(toplevel):
                l2 = []
                for k, value in enumerate(mediumlevel):
                    l2.append(value * tensor_2[i][j][k])
                l1.append(l2)
            tensor_result.append(l1)
    else:
        raise Exception('Debe definir un algoritmo en el parametro alg')

    return np.array(tensor_result), tensor_1.shape, tensor_2.shape

U = np.array([
        [[1,2,3], [4,5,6], [7,8,9]],
        [[11,12,13], [14,15,16], [17,18,19]],
        [[21,22,23], [24,25,26], [27,28,29]]
    ])

print('\n\n(2) producto Hadamard: ', tensors_product(U, U, alg='Hadamard'))
print('\n\n(2) producto de los tensores: ', tensors_product(U, U, alg='tensor_product'))


"""
    3) implemente una función que reciba por parámetro dos matrices A y B y verifique si B es
    la descomposición de Cholesky de A
"""


def check_Cholesky(matrx1, matrx2):
    return (np.round(matrx2.dot(matrx2.T)) == matrx1).all()

M = np.array([[2., 1., 1.], [1., 2., 1.], [1., 1., 2.]])
# nice Cholesky of M
G = cholesky(M)
# bad Cholesky of M
# G = np.array([[2.41421356, 0., 0.], [0.70710678, 1.22474487, 0.], [0.70710678, 0.40824829, 1.15470054]])
print('\n\n(3) es descomposición de Cholesky? ', check_Cholesky(M, G))


"""
    4) implemente una función que reciba por parámetro una matriz y un vector y verifique si el
    vector es vector propio de la matriz
"""


def is_eigenvector(matrx, v):
    values, vectors = eig(matrx)
    _decimals = 10
    for i in range(0, v.shape[0]):
        if (np.around(matrx @ vectors[:, i], decimals=_decimals) == np.around(v, decimals=_decimals)).all():
            return True
    return False


W = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
values, vectors = eig(W)
# nice vector
v = W @ vectors[:, 1]  # may change index 0, 1, or 2
# non-apropiate vector
# v = np.array([ -99999.73863537,  -8.46653421, -13.19443305])
print('\n\n(4) es vector propio de la matriz?', is_eigenvector(W, v))