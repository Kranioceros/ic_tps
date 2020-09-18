# numpy para matematica matricial y numeros aleatorios
# matplotlib para graficar
import numpy as np
import matplotlib.pyplot as plt
from utils import extender, signo, const

# Este archivo define una clase para una red neuronal multicapa, con la funcion
# sigmoidea como funcion de activacion, entrenable con el metodo de retropropagacion
# del error


class RedMulticapa:
    # Recibe mandatoriamente una lista `arq` de por lo menos 2 elementos que
    # especifica la arquitectura de la red neuronal
    # El primer elemento especifica el numero de entradas a la red neuronal
    # (sin incluir el sesgo). Los siguientes especifican el numero de neuronas de cada capa
    # Por defecto se inicializan los pesos en 0, pero en `interv_rand` puede darse una tupla
    # que especifique el intervalo del cual llenar aleatoriamente los pesos
    # `activ` es la funcion de activacion de las neuronas
    # `deriv` es la derivada de la funcion de activacion, utilizada en el entrenamiento
    # Por defecto, el metodo de entrenamiento es el de Widrow-Hoff con la funcion signo
    def __init__(self, arq, interv_rand=None, activ=signo, deriv=const):
        if len(arq) < 2:
            print(
                "Error: arq debe tener al menos 2 elementos (entradas de la red y una capa)")
            return

        # Se inicializa un generador de numeros aleatorios
        self.rng = np.random.default_rng()

        # Lista con las matrices de peso de cada capa
        self.ws = []

        # Se definen las funciones de activacion y derivacion
        self.activ = activ
        self.deriv = deriv

        # Ahora construimos las matrices de peso de cada capa. Para eso,
        # recorremos la lista `arq` de principio a fin, utilizando el numero
        # de neuronas de la capa anterior y la actual para definir las dimensiones
        # de la matriz de pesos de la capa actual

        # El numero de entradas en la capa actual
        entradas = arq.pop(0)

        # Creamos las matrices de pesos
        for nro_neuronas in arq:
            if interv_rand is None:
                w = np.zeros((nro_neuronas, entradas+1))
            else:
                (a, b) = interv_rand
                w = (b - a) * self.rng.random((nro_neuronas, entradas+1)) + a

            self.ws.append(w)
            entradas = nro_neuronas

    # Aleatoriza los valores de los pesos de todas las capas, tomando valores
    # del `interv_rand` recibido como parametro (sustituyendo los anteriores)
    def aleatorizar(self, interv_rand):
        (a, b) = interv_rand
        nuevo_ws = []
        for w in self.ws:
            nuevo_w = (b - a) * self.rng.random(w.shape) + a
            nuevo_ws.append(nuevo_w)
        self.ws = nuevo_ws

    # Evalua la red neuronal para un patron x dado
    # `x`: vector 1D con el patron, SIN el sesgo
    def evaluar(self, x):
        if x.ndim != 1:
            print("evaluar: error de dimension en `x`")
        # Salida de la capa actual (convertimos en matriz de Nx1)
        y = x[:, np.newaxis]
        # Propagamos hacia adelante
        for w in self.ws:
            # Extendemos la salida de la capa anterior para el sesgo
            y_ext = extender(y, dim=0)
            # Calculamos activacion lineal y salida de la capa
            v = w @ y_ext
            y = self.activ(v)
        return y

    # Entrena la red neuronal con los patrones completos y sus respectivas salidas en `datos_entr`
    def entrenar(self, datos_entr, max_epocas=100, umbral_err=0.05, coef_apren=0.1):
        # TODO
        pass
