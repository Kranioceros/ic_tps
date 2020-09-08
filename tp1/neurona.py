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
    # Si se activa la opcion `guardar_pesos`, tambien devuelve los pesos para cada paso
    # `datos_entr` contiene entradas y sus salidas correspondientes (entrada extendida)
    # `max_epocas` define el numero maximo que se pueden repetir todos los patrones
    # `umbral_err` es el error relativo umbral a utilizar para finalizar el algoritmo
    # `coef_apren` es el coeficiente de aprendizaje
    def entrenar(self, datos_entr, max_epocas=500, umbral_err=0.05, coef_apren=0.1, guardar_pesos=False):
        (nro_patrones, entrada_ext) = datos_entr.shape

        if entrada_ext != self.dimension+1:
            print("La dimension de datos_entr no corresponde con la neurona")
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
            print("La dimension de x no corresponde con la neurona")
            return

        return np.vectorize(self.fn_activacion)(x @ self.w)

    # Toma `x` como entrada de la neurona e `yd` como salida deseada de la misma
    # Devuelve la salida de la neurona `y` y tambien su error `err`

    def probar(self, x, yd):
        y = self.evaluar(x)
        err = yd - y
        return (y, err)

    # Realiza un grafico de la recta de la neurona (si usa 3 entradas)
    # Si `show` es True, se llama a show al final de la funcion
    # Si `pause` es True, se llama a pause al final de la funcion, con tiempo `time`
    def graficar3(self, show=True, pause=False, time=3):
        w = self.w
        plt.axline((0, w[0]/w[2]), slope=-w[1]/w[2])
        plt.axis([-1.1, 1.1, -1.1, 1.1])
        if show:
            plt.show()
        elif pause:
            plt.pause(time)
        else:
            pass
