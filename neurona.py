# numpy para matematica matricial y numeros aleatorios
# matplotlib para graficar
import numpy as np
import matplotlib.pyplot as plt

from utils import signo

# Recibe opcionalmente `dim`, que define el numero de entradas de la neurona,
# INCLUYENDO el sesgo. Inicializa en cero los w.
# Recibe opcionalmente `fn_activ`, que define la funcion de activacion de la neurona.
# Por defecto es la funcion signo


class Neurona:
    def __init__(self, dim=0, fn_activ=signo):
        self.dimension = dim
        self.fn_activacion = fn_activ
        self.w = np.zeros(dim)
        # Inicializamos generador de numeros aleatorios
        self.rng = np.random.default_rng()

    # Genera un vector de pesos con cada peso en (a, b), con b > a
    def aleatorizar(self, a, b):
        for i in range(0, self.dimension):
            self.w[i] = (b-a) * self.rng.random() + a

    # Entrena la neurona, devolviendo el el error final y el numero de epocas usadas
    # `datos_entr` contiene entradas y sus salidas correspondientes (entrada extendida)
    # `max_epocas` define el numero maximo que se pueden repetir todos los patrones
    # `err_rel` es el error relativo umbral a utilizar para finalizar el algoritmo
    # `coef_apren` es el coeficiente de aprendizaje
    def entrenar(self, datos_entr, max_epocas=500, err_rel=0.05, coef_apren=0.1):
        (nro_patrones, entrada_ext) = datos_entr.shape

        if entrada_ext != self.dimension+1:
            print("La dimension de datos_entr no corresponde con la neurona")
            return

        # Entrada: todas las columnas menos la ultima
        x = datos_entr[:, :-1]
        # Salida deseada: la ultima columna
        yd = datos_entr[:, -1]

        # Inicializamos epoca y error
        epoca = 0
        err_r = np.ones((nro_patrones, 1)) * err_rel * 10
        while epoca < max_epocas and any(err_r > err_rel):
            for i in range(nro_patrones):
                # Calculamos la salida de la neurona y su error
                yi = self.fn_activacion(self.w @ x[i, :])
                err_r[i] = abs((yd[i] - yi) / yd[i])
                self.w += (coef_apren / 2) * (yd[i] - yi) * x[i, :]
            epoca += 1

        return (err_r, epoca)

    # Estimula la neurona con la entrada x y devuelve su salida
    def evaluar(self, x):
        (_filas, cols) = x.shape
        if cols != self.dimension:
            print("La dimension de x no corresponde con la neurona")
            return

        return np.vectorize(self.fn_activacion)(x @ self.w)
