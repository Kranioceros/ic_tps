import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Funciones de activacion


def signo(v):
    return (v <= 0) * (-1) + (v > 0) * 1

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

# Derivadas de funciones de activacion

# La funcion const devuelve una matriz o vector de igual dimension cuyos
# valores son todos `a`. Es util para el metodo de Widrow-Hoff, donde se
# asume que la funcion de activacion es simplemente la activacion lineal


def const(v, a=1):
    return v * 0 + a

# Extiende una matriz de m x n con una columna al inicio cuyos valores son -1
# Si es un vector, simplemente agrega un -1 al principio
# `dim` define si se agrega una columna con -1 o una fila (la dimension)


def extender(x, dim=1):
    if x.ndim == 1:
        return np.concatenate([-np.ones(1), x])
    elif dim == 1:
        (fil, _cols) = x.shape
        col0 = np.ones((fil, 1)) * (-1)
        return np.hstack((col0, x))
    else:
        (_fil, cols) = x.shape
        fil0 = np.ones((1, cols)) * (-1)
        return np.vstack((fil0, x))


# Funcion sigmoidea simetrica para utilizar en perceptrones multicapa. Opera sobre un vector


def sig(x):
    return 2*np.reciprocal(1 + np.exp(-x)) - 1

# Crea `n` particiones con un porcentaje `p` de patrones de entrenamiento y 1-`p` de pruebas
# `p` es un valor entre 0 y 1
# La entrada de datos es `m`
# Devuelve un arreglo de `n` particiones
# Cada particion es una tupla de indices de la forma:
# (idx_entr, idx_prueba)
# donde cada idx es un vector con los indices de las filas correspondientes en `m`
# Si random es True, los vectores de indices son elegidos aleatoriamente


def particionar(m, n, p, random=False):
    (fils, _cols) = m.shape
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

# Toma una matriz `m` de dimension M x N, donde cada fila es una entrada de `f`
# La funcion utiliza las particiones `part` para evaluar `f` para cada particion
# `part` es un vector de indices de filas
# Devuelve un vector columna con el resultado de aplicar `f` a la particion


def part_apply(m, part, f):
    # Aplicamos la funcion a las filas correspondientes de la particion actual
    return np.apply_along_axis(f, 1, m[part, :])

# Configura el axes para que sea una grilla cuadrada de lado `tam`
# Agrega ejes y se asegura que se mantenga cuadrada


def configurar_grilla(ax, tam=1.3):
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)

    # Configuraciones especificas para 2D o 3D
    if isinstance(ax, Axes3D):
        ax.set_zlim(-1.3, 1.3)
    else:
        ax.axhline(y=0, c='grey')
        ax.axvline(x=0, c='grey', lw=1)
        ax.grid(ls='dashed')
        ax.set_aspect('equal', adjustable='box')
