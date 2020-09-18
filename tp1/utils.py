import numpy as np
import matplotlib.pyplot as plt

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

# Toma pesos de una neurona, datos de entrenamiento y visualiza de forma animada
# como ajusta sus parametros. Para eso crea una grafica, la anima y la cierra.
# Funciona para casos 2D (neurona con 2 entradas)
# `pesos` es una matriz con los pesos de las neuronas en cada paso de entrenamiento
# `max_pasos` es el numero de pasos antes de terminar la animacion
# `t` es el numero de segundos entre paso y paso
# `titulo` titulo de la grafica
# `labels` es una tupla con los labels del eje x e y


def visualizar2D(pesos, datos_entr, max_pasos=1000, t=0.0001, titulo=None, labels=('x', 'y')):
    (_fil, col) = datos_entr.shape

    if len(pesos[0]) != 3 or col != 3:
        print("Error de dimensiones en visualizar")

    n = min(len(pesos), len(datos_entr), max_pasos)

    # Crea una figura y le agrega un axes
    fig = plt.figure()
    ax = fig.add_subplot()

    for i in range(n):
        # Titulo y labels
        if titulo is not None:
            ax.set_title(titulo)
        (xlabel, ylabel) = labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Punto
        w = pesos[i]
        x1 = datos_entr[i, 0]
        x2 = datos_entr[i, 1]

        if datos_entr[i, 2] == 1:
            ax.scatter(x1, x2, c='b')
        else:
            ax.scatter(x1, x2, c='r')

        # Recta
        ax.axline((0, w[0]/w[2]), slope=-w[1]/w[2])

        # Configuracion
        configurar_grilla(ax)

        # Mostramos, esperamos y borramos
        plt.pause(t)
        plt.cla()

    # Cerramos la figura
    plt.close(fig)


# Configura el axes para que sea una grilla cuadrada de lado `tam`
# Agrega ejes y se asegura que se mantenga cuadrada
def configurar_grilla(ax, tam=1.3):
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.axhline(y=0, c='grey')
    ax.axvline(x=0, c='grey', lw=1)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(ls='dashed')
