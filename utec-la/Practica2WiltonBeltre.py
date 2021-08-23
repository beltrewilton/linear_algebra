import numpy as np
import timeit as t
from random import randint
from itertools import permutations
from numpy.linalg import matrix_rank

"""
   1) Implemente una función que reciba una matriz como parámetro e indique por
    pantalla si la matriz es:
    • Matriz cuadrada
    • Matriz simétrica
    • Matriz triangular
    • Matriz diagonal
    • Matriz de identidad
    • Matriz ortogonal
"""

"""
 > Custom helpers functios
"""


def transpose(matx):
    return np.array([matx[:, r] for r in range(matx.shape[1])])


def build_test_triangular(n):
    matx = np.random.randint(1, 10, size=(n, n))
    for idx in range(1, matx.shape[0]):
        for idx2 in range(0, idx):
            matx[idx, idx2] = 0
    return matx


def check_triangular(matx):
    for idx in range(1, matx.shape[0]):
        for idx2 in range(0, idx):
            if matx[idx, idx2] != 0:
                return False
    return True


def is_triangular(matx):
    return check_triangular(matx) or check_triangular(transpose(matx))


def build_test_diagonal(n):
    test = np.random.randint(1, 10, size=(n))
    diag = np.zeros((n, n))
    for idx in range(0, n):
        for idx2 in range(idx, idx + 1):
            diag[idx, idx2] = test[idx]
    return diag


def is_diagonal(matx):
    size = matx.shape[0]
    m = np.copy(matx) # tricky
    zeros = np.zeros((m.shape[0], m.shape[1]))
    for idx in range(0, size):
        for idx2 in range(idx, idx + 1):
            m[idx, idx2] = 0
    return (m == zeros).all()


def build_test_identity(n):
    diag = np.zeros((n, n))
    for idx in range(0, n):
        for idx2 in range(idx, idx + 1):
            diag[idx, idx2] = 1
    return diag


def is_identity(mtx):
    ident = build_test_identity(mtx.shape[0])
    return (mtx == ident).all()


def matx_dot_product(matx1, matx2):
    if matx1.shape[1] != matx2.shape[0]:
        raise Exception('[n x m] debe ser igual a [m x k]')

    matx_result = []
    for idx1 in range(matx1.shape[0]):
        tmp = []
        for idx2 in range(matx2.shape[1]):
            tmp.append(sum(matx1[idx1] * matx2[:, idx2]))
        matx_result.append(tmp)
    return np.array(matx_result)


def is_ortogonal(matx1, matx2):
    return (not matx_dot_product(matx1, matx2).any()) or (not matx_dot_product(matx1, transpose(matx2)).any())


def beauti(matx1):
    raw = ''
    for i,m in enumerate(matx1):
        raw += '{}\n'.format(m).replace('[', '|').replace(']', '|')
    return raw

def vertical(matx1, matx2):
    raw = ''
    for i,m in enumerate(matx1):
        raw += '{}    {}\n'.format(m, matx2[i]).replace('[', '|').replace(']', '|')
    return raw


def main_testing_func(matx, matx2=None):
    if matx.shape[0] == matx.shape[1]:
        print('Ja! esta es una matrix cuadrada')
        if (transpose(matx) == matx).all():
            print('Oh! ademas es simetrica!')
    if is_triangular(matx):
        print('Confirmado, es triangular.')
    if is_diagonal(matx):
        print('Es diagonal.')
    if is_identity(matx):
        print('Es de identidad')
    if matx2 is not None:
        if is_ortogonal(matx, matx2):
            print('Son ortogonales, increible!')


H = np.array([[5, 6, 7], [6, 7, 5], [7, 5, 6]])
print('test1: \n{}'.format(beauti(H)))
main_testing_func(H)

T = build_test_triangular(5)
Tn = [[4, 4, 8, 3, 4], [0, 9, 1, 7, 5], [0, 0, 1, 2, 9], [999, 0, 0, 2, 9], [0, 0, 0, 0, 2]]
print('\ntest2: \n{}'.format(beauti(T)))
main_testing_func(T)

Dx = np.array([[5, 0, 0], [0, 2, 0], [0, 1, 2]])
D = build_test_diagonal(3)
print('\ntest3: \n{}'.format(beauti(D)))
main_testing_func(D)

I = build_test_identity(3)
In = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
print('\ntest4: \n{}'.format(beauti(In)))
main_testing_func(In)

A1 = np.array([[1,2,3], [2,4,6], [1,5,9]])
A2 = np.array([[0,0,0], [0,0,0], [0,0,0]])
print('\ntest5: \n{}'.format(vertical(A1, A2)))
main_testing_func(A1, A2)


"""
    2) Implemente una función que reciba dos matrices como parámetro e indique por
        pantalla si:
        • Una es la Transpuesta de la otra
        • Una es la Inversa de la otra
"""

def is_traspose_mutual(matx1, matx2):
    if matx1.shape[0] != matx2.shape[1]:
        raise Exception('[n x m] debe ser igual a [m x k]')
    return (matx1 == transpose(matx2)).all()


def is_inverse(matx1, matx2):
    if not ((matx1.shape[0] == matx1.shape[1]) and (matx2.shape[0] == matx2.shape[1])):
        raise Exception('Ambas matrices deben ser cuadradas.')
    return (matx1 == np.round(np.linalg.inv(matx2))).all()


T1 = np.array([[3, 4, 5], [1, 2, 4],  [6, 7, 8],  [6, 2, 5], [1, 3, 2]])
T2 = np.array([[3, 1, 6, 6, 1],  [4, 2, 7, 2, 3], [5, 4, 8, 5, 2]])
print('\ntest6 (Una es la Transpuesta de la otra?): \n{}\n{}'.format(beauti(T1), beauti(T2)))
print('Respuesta: {}\n'.format(is_traspose_mutual(T1, T2)))


M1 = np.array([[2., 3., 1.],  [6., 4., 3.],  [2., 6., 3.]])
I1 = np.array([[3.00000000e-01,  1.50000000e-01, -2.50000000e-01],
               [6.00000000e-01, -2.00000000e-01,  4.75809868e-17],
               [-1.40000000e+00,  3.00000000e-01,  5.00000000e-01]])
print('test7 (Una es inversa de la otra?)\nMatriz\n{}\nInversa\n{}'.format(beauti(M1), beauti(I1)))
print('Respuesta: {}\n'.format(is_inverse(M1, I1)))


"""
    3) Implemente una función que calcule paso a paso el determinante de una matriz de
    3x3 mediante su algoritmo preferido y compare su resultado con el resultado con la
    función det.
    > fuente: https://en.wikipedia.org/wiki/Determinant
"""

from itertools import permutations


def consecutive(row):
    return (row[1]-row[0] == 1) or (row[2]-row[1] == 1)


def Leibniz_det(matx):
    perm = list(permutations([1, 2, 3]))
    return sum([((1 if consecutive(a) else -1) * matx[0][a[0]-1]) * matx[1][a[1]-1] * matx[2][a[2]-1] for a in perm])


test_matrx = np.random.randint(2,7, size=(3,3))
print('random 3x3')
print(beauti(test_matrx))
print('In my own: {}'.format(Leibniz_det(test_matrx)))
print('NumPy: {}\n'.format(np.linalg.det(test_matrx)))


"""
    4) Cree una matriz de 5x5, obtenga su traza y muestre por pantalla tanto la matriz como
    la traza obtenida.
"""


def trace_matx():
    matx = np.random.randint(1,9, size=(5,5))
    print('test9\n{}'.format(beauti(matx)))
    tmp = []
    for idx in range(0, matx.shape[0]):
        for idx2 in range(idx, idx + 1):
            tmp.append(matx[idx, idx2])

    print('Traza obtenida:\n{}\n'.format(np.array(tmp)))

trace_matx()


"""
    5) Cree una matriz de ceros de 2x2 y obtenga su rango, muestre la matriz y el rango
    obtenido por pantalla.
"""
zeros = np.zeros((2,2))
print('test10\nMatriz de ceros\n{}\nRango: {}'.format(beauti(zeros), matrix_rank(zeros)))