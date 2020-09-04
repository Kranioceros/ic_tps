import numpy as np

# Funciones de activacion


def signo(n):
    if n < 0:
        return -1
    else:
        return 1

# `alfa` es la pendiente de la rampa, `a` define los tramos de la funcion
# Devuelve una funcion f de un solo parametro con alfa y a


def rampa(alfa, a):
    def f(n):
        if n < (-a):
            return -1
        elif n < a:
            return alfa*n
        else:
            return 1
    return f

# Extiende una matriz de m x n con una columna al inicio cuyos valores son -1


def extender(m):
    (fil, _col) = m.shape
    col0 = np.ones((fil, 1)) * (-1)
    return np.hstack((col0, m))
