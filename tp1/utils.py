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

# Crea `n` particiones con un porcentaje `p` de patrones de entrenamiento y 1-`p` de pruebas
# `p` es un valor entre 0 y 1
# La entrada de datos es `m`
# Devuelve un arreglo de `n` particiones
# Cada particion es una tupla de indices de la forma:
# (idx_entr, idx_prueba)
# donde cada idx es un vector con los indices de las filas correspondientes en `m`
# Si random es True, los vectores de indices son elegidos aleatoriamente


def particionar(m, n, p, random=False):
    (fils, cols) = m.shape
    particiones = []

    for _i in range(n):
        # Idx contiene los indices de todas las filas
        idx = np.arange(fils)
        # Los mezclamos si random=True
        if random:
            np.random.shuffle(idx)
        # Calculamos el numero de patrones de entrenamiento en base a `p`
        Ne = int(np.floor(fils*p))
        # Agregamos los indices al arreglo de particiones
        idx_entr = idx[:Ne]
        idx_prueba = idx[Ne:]

        particiones.append((idx_entr, idx_prueba))

    return particiones
