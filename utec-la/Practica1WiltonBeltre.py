import numpy as np
import timeit as t

"""
(1) Usando exclusivamente las funciones abordadas en la Lecture 1 (zeros, ones, hstack y
    vstack), implementar una función que cree la siguiente matriz e imprimir su tamaño y forma
    por pantalla:
                    1. 1. 0 0
                    1. 1. 0 0
                    0 0 1. 1.
                    0 0 1. 1.
"""


def crea_matriz():
    m = np.hstack(np.array([np.ones(2), np.zeros(2)]))
    return np.array([m, m, m[::-1], m[::-1]])


m = crea_matriz()
print("Ejercicio #1")
print(m)
print('Tamano de m: {}\nForma de m: {}\n\n'.format(m.size, m.shape))


"""
(2) Implementar una función que reciba dos parámetros enteros (n y m) y retorne una matriz
    bidimensional de números enteros entre 0 y 100 aleatorios (tip : random). Ejemplo con n=3
    y m=2 debería retornar aleatoriamente:
                                            4 98
                                            55 65
                                            1 88
"""


def random_matrix(n, m):
    return np.random.randint(0, 100, size=(n, m))


print("Ejercicio #2")
print('Matrix aleatoria nxm de numeros enteros entre (0,100)\n{}\n\n'.format(random_matrix(3, 2)))


"""
(3) Implementar una función que reciba dos parámetros : una matriz bidimensional de n x m de
    numeros enteros (tipo de datos ndarray) y un escalar , y realice manualmente la difusión de
    la matriz con el escalar.
    Manualmente significa que no debe usar la operación suma de Numpy (entre ndarray y un
    escalar) sino que debe recorrer cada uno de los elementos de la matriz y retornar una
    nueva matriz sumando enteros al escalar.
"""


def diffusion(m, s):
    f = []
    for i in m:
        a = []
        for x in i:
            a.append(x+s)
        f.append(a)
    return np.array(f)

escalar = 4
m = random_matrix(5, 7)
print("Ejercicio #3")
print('Matriz aleatoria (5x7) generada: \n{}\n'.format(m))
print('Elemento escalar: {}'.format(escalar))
print('Matriz resultante (cada elemento+{}): \n{}\n\n'.format(escalar, diffusion(m, escalar)))


"""
(4) Utilizando las funciones anteriores compare el tiempo ejecución de su función implementada
    en el ejercicio anterior con la función difusión de NumPy (Lecture 3.2).
    Compare los tiempos de ejecución de cada una para realizar la suma de una matriz
    aleatoria con un escalar para matrices de 10 x10 , 1.000 x 1.000 y 10.000 x 10.000
"""

print("Ejercicio #4")
def benchmark(n, m):
    global randm
    randm = random_matrix(n, m)
    execution_time = t.timeit('diffusion(randm, 2)', 'from __main__ import  diffusion, randm', number=1)
    print('{}x{} myfunction execution_time: {}'.format(n, m, execution_time))
    execution_time = t.timeit('randm+2', 'from __main__ import diffusion, randm', number=1)
    print('{}x{} numpy      execution_time: {}\n'.format(n, m, execution_time))


benchmark(10, 10)
benchmark(100, 100)
benchmark(1000, 1000)
benchmark(10000, 10000)

