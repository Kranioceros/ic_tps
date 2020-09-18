# numpy para matematica matricial y numeros aleatorios
# matplotlib para graficar
import numpy as np
import matplotlib.pyplot as plt

from utils import signo, part_apply, configurar_grilla

# Recibe opcionalmente `dim`, que define el numero de entradas de la neurona,
# INCLUYENDO el sesgo. Inicializa en cero los w.
# Recibe opcionalmente `fn_activ`, que define la funcion de activacion de la neurona.
# Por defecto es la funcion signo
# `aleatorizar` define el intervalo inicial de los pesos (por defecto no se inicializan)


class Neurona:
    def __init__(self, dim=0, fn_activ=signo, aleatorizar=()):
        self.dimension = dim
        self.fn_activacion = fn_activ
        self.w = np.zeros(dim)
        # Inicializamos generador de numeros aleatorios
        self.rng = np.random.default_rng()
        if aleatorizar != ():
            (a, b) = aleatorizar
            self.aleatorizar(a, b)

    # Genera un vector de pesos con cada peso en (a, b), con b > a
    def aleatorizar(self, a, b):
        for i in range(0, self.dimension):
            self.w[i] = (b-a) * self.rng.random() + a

    # Entrena la neurona, devolviendo el el error final y el numero de epocas usadas
    # Si se activa la opcion `guardar_pesos`, tambien devuelve los pesos para cada paso
    # `datos_entr` contiene entradas y sus salidas correspondientes (entrada extendida)
    # `max_epocas` define el numero maximo que se pueden repetir todos los patrones
    # `umbral_err` es el error relativo umbral a utilizar para finalizar el algoritmo
    # `coef_apren` es el coeficiente de aprendizaje
    def entrenar(self, datos_entr, max_epocas=100, umbral_err=0.05, coef_apren=0.1, guardar_pesos=False):
        (nro_patrones, entrada_ext) = datos_entr.shape

        if entrada_ext != self.dimension+1:
            print(
                f'La dimension de datos_entr({datos_entr.shape}) no corresponde con la neurona({self.dimension})')
            return

        # Entrada: todas las columnas menos la ultima
        x = datos_entr[:, :-1]
        # Salida deseada: la ultima columna
        yd = datos_entr[:, -1]

        # Inicializamos epoca, error y pesos
        epoca = 0
        err = np.ones((nro_patrones, 1)) * umbral_err * 10
        pesos = []
        while epoca < max_epocas and any(err > umbral_err):
            for i in range(nro_patrones):
                if guardar_pesos:
                    pesos.append(list(self.w))
                # Calculamos la salida de la neurona y su error
                yi = self.fn_activacion(self.w @ x[i, :])
                err[i] = abs((yd[i] - yi) / yd[i])
                self.w += (coef_apren / 2) * (yd[i] - yi) * x[i, :]
            epoca += 1

        return (err, epoca, pesos)

    # Estimula la neurona con la entrada x y devuelve su salida
    def evaluar(self, x):
        if x.ndim > 1 or x.size != self.dimension:
            print(
                f'La dimension de x({x.shape}) no corresponde con la neurona({self.dimension})')
            return
        y = self.fn_activacion(x @ self.w)
        return y

    # Toma `x` como entrada de la neurona e `yd` como salida deseada de la misma
    # Devuelve la salida de la neurona `y` y tambien su error `err`

    def probar(self, x, yd):
        y = self.evaluar(x)
        err = yd - y
        return (y, err)

    # Realiza un grafico de la recta de la neurona (si usa 3 entradas)
    # `ax` es un objeto de tipo Axes, sobre el cual se grafica
    def graficar2(self, ax):
        w = self.w
        ax.axline((0, w[0]/w[2]), slope=-w[1]/w[2])

# Toma un set de datos y una particion de los mismos
# A partir de eso entrena una neurona y calcula el error promedio para cada particion
# `m` es la matriz de datos (contiene la entrada extendida y la salida deseada)
# `parts` es un arreglos de tuplas de la forma:
#  [(idx_entr, idx_prueba)]
#  donde idx_entr e idx_prueba son vectores de indices de filas de m
# `init_opts` es un diccionario con las opciones para inicializa la neurona (argumentos de constructor)
# `entr_opts` es un diccionario con las opciones para entrnar la neurona (argumentos de entrenamiento)
# Devuelve la neurona entrenada de cada particion junto con su error promedio


def validacion_cruzada(m, parts, init_opts, entr_opts):
    # Entradas de las neuronas (todas menos la ultima columna)
    x = m[:, :-1]
    # Salidas deseadas de las neurona (la ultima columna)
    yd = m[:, -1]
    # Inicializamos arreglos de neuronas y vector de errores
    neuronas = []
    errores = []

    for (part_entr, part_prueba) in parts:
        # Creamos una neurona y aleatorizamos sus pesos
        an = Neurona(**init_opts)
        # La entrenamos con la particion de entrenamiento
        an.entrenar(m[part_entr, :], **entr_opts)
        # La evaluamos con la particion de pruebas
        y = part_apply(x, part_prueba, an.evaluar)
        # Calculamos el error
        err = np.average(np.abs(yd[part_prueba] - y))
        # Guardamos la neurona y el error
        neuronas.append(an)
        errores.append(err)
        # Graficamos
        # an.graficar3(show=True)

    return (neuronas, np.array(errores))

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
