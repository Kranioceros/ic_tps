import numpy as np
import math

# Funciones de activacion


def signo(n):
    return (n<=0)*(-1) + (n>0)*1

def dsigno(x):
    return np.ones(x.shape)
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

#Activacion por sigmoide
def sigmoid(x):
    return 2*np.reciprocal(1 + np.exp(-x)) - 1

#Derivada sigmoide simétrica
def dsigmoid(x):
    return (1+x)*(1-x)*0.5

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


def particionar_k_out(m, k):
    (fils, cols) = m.shape
    particiones = []

    # Cantidad de particiones
    n = int(np.floor(fils/k))

    # Idx contiene los indices de todas las filas
    idx = np.arange(fils)
    
    rango = 0

    for _i in range(n):
        # Agregamos los indices al arreglo de particiones
        idx_prueba = idx[rango:k+rango]
        # Diferencia entre conjuntos (set) y devuelto como array
        idx_entr = np.array(list(set(idx)-set(idx_prueba)))

        particiones.append((idx_entr, idx_prueba))

        rango += k

    return particiones

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

# Toma una matriz `m` de dimension M x N, donde cada fila es una entrada de `f`
# La funcion utiliza las particiones `part` para evaluar `f` para cada particion
# `part` es un vector de indices de filas
# Devuelve un vector columna con el resultado de aplicar `f` a la particion


def part_apply(m, part, f):
    # Aplicamos la funcion a las filas correspondientes de la particion actual
    return np.apply_along_axis(f, 1, m[part, :])

def convert_to_one_dimension(x):
    x1_prom = np.mean(x[:,0])
    x2_prom = np.mean(x[:,1])
    y = x[:,-1]
    x_new = []

    for _i in range(len(x[:,0])):
        x_new.append([math.sqrt((x1_prom-x[_i][0])**2+(x2_prom-x[_i][1])**2),y[_i]])
    return np.array(x_new)

def WinnerTakesAll(x):
    max_idx = np.argmax(x)
    v_x = np.ones(len(x))*-1
    v_x[max_idx] = 1
    return v_x

def Calcular_Area(x):
    datos_area = np.zeros((x.shape[0],5))

    for _p in range(x.shape[0]):
        datos_area[_p, 0] = x[_p, 0]*x[_p, 1]
        datos_area[_p, 1] = x[_p, 2]*x[_p, 3]
        datos_area[_p, 2] = x[_p, 4]
        datos_area[_p, 3] = x[_p, 5]
        datos_area[_p, 4] = x[_p, 6]
    
    return datos_area

#Promedia alto y ancho de cefalo y petalo
def Promediar(x):
    datos_prom = np.zeros((x.shape[0],5))

    for _p in range(x.shape[0]):
        datos_prom[_p, 0] = x[_p, 0]+x[_p, 2]/2
        datos_prom[_p, 1] = x[_p, 1]+x[_p, 3]/2
        datos_prom[_p, 2] = x[_p, 4]
        datos_prom[_p, 3] = x[_p, 5]
        datos_prom[_p, 4] = x[_p, 6]

    return datos_prom

def Normalizar(x):
    datos_norm = np.zeros(x.shape)

    for _p in range(x.shape[0]):
        max_idx = np.argmax(x[_p,:])
        datos_norm[_p, :-3] = x[_p, :-3]/x[_p, max_idx]
        datos_norm[_p, -3:] = x[_p, -3:]

    return datos_norm

def gaussiana(x,media,sigma):
    dist = x-media
    distCuad = np.dot(dist,dist)
    return np.exp(-(distCuad)/(2*(sigma**2)))