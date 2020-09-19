# numpy para matematica matricial y numeros aleatorios
# matplotlib para graficar
import numpy as np
import matplotlib.pyplot as plt
from utils import extender, sig, dsig

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
    # Por defecto, el metodo de entrenamiento es con la funcion sigmoidea y su derivada
    def __init__(self, arq, interv_rand=None, activ=sig, deriv=dsig):
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
    # `por_capa`: si se activa, devuelve la salida de cada capa (incluyendo la entrada)
    # Devuelve un vector 1D con la salida, o si `por_capa` esta activado, una lista de vectores 1D
    def evaluar(self, x, por_capa=False):
        if x.ndim != 1:
            print("evaluar: error de dimension en `x`")
        # Vector con las salidas y activacion lineal de cada capa
        ys = []
        # Salida de la capa actual (convertimos en matriz de Nx1)
        y = x
        ys.append(y)
        # Propagamos hacia adelante
        for w in self.ws:
            # Extendemos la salida de la capa anterior para el sesgo
            y_ext = extender(y, dim=0)
            # Calculamos activacion lineal y salida de la capa
            v = w @ y_ext
            y = self.activ(v)
            ys.append(y)

        if(por_capa):
            return ys
        else:
            return ys[-1]

    # Entrena la red neuronal con los patrones completos y sus respectivas salidas en `datos_entr`
    def entrenar(self, datos_entr, max_epocas=10, umbral_err=0.1, coef_apren=0.1):
        (nro_patrones, cols) = datos_entr.shape
        if cols != self.ws[0][0, :].size:
            print(
                f'entrenar: error de dimension, entrada con {cols} columnas y red con {self.ws[0][0,:].size} entradas')

        x = datos_entr[:, :-1]
        yd = datos_entr[:, -1]
        yd = yd.reshape(-1, 1)       # Convertimos en matriz de Nx1

        print("### ENTRENAR ###")

        # Por cada epoca
        for _epoca in range(max_epocas):
            # Por cada patron
            for i in range(nro_patrones):
                print("### COMIENZA PATRON ###")
                # Evaluamos la salida de cada capa para el patron actual
                ys = self.evaluar(x[i], por_capa=True)

                # Inicializamos el error retropagado. Para la ultima capa, se usa
                # el error de la red. A partir de `er` se calcula el ajuste de los
                # pesos en cada capa
                er = (yd[i] - ys[-1]).reshape(-1, 1)

                # Algunos calculos tienen anotaciones para entender las dimensiones de las
                # matrices y vectores. Aca esta la referencia:
                # `N2`: Nro. neuronas capa actual, `N1`: Nro. neuronas capa anterior

                # Ajustamos los pesos de cada capa en base al `er` provisto por la
                # siguiente capa
                # `w`  : Pesos de la capa actual    (N2xN1+1)
                # `y`  : Salida de la capa actual   (N2x1)
                # `y_a`: Salida de la capa anterior (N1x1)
                for w, y, y_a in zip(reversed(self.ws), reversed(ys), reversed(ys[:-1])):
                    # Transformamos y e y_a en vectores columna
                    y = y.reshape(-1, 1)
                    y_a = y_a.reshape(-1, 1)

                    # print(f'yd[i]: {yd[i]}')
                    print(f'y: {y}, shape = {y.shape}')
                    print(f'y_a: {y_a}, shape = {y_a.shape}')
                    print(f'er: {er}')
                    print(f'w: {w}')

                    # Calculamos el gradiente de error local instantaneo
                    # (N2x1) * (N2x1) = (N2x1)
                    el = er*self.deriv(y)

                    print(f'el: {el}, shape = {el.shape}')

                    # dw es el ajuste de los pesos de la capa actual
                    #  alfa * (N2x1) @ (1xN1+1) = (N2x1) @ (1xN1+1) = (N2xN1+1)
                    dw = coef_apren * el @ extender(y_a.T)

                    print(f'dw: {dw}, shape = {dw.shape}')
                    # print(f'w: {w}, shape = {w.shape}')

                    # Ajustamos los pesos
                    # (N2xN1+1) - (N2xN1+1) = (N2xN1+1)
                    w += dw

                    # Calculamos el error propagado hacia la capa anterior
                    # (N1xN2) @ (N2x1) = (N1x1)
                    er = w[:, 1:].T @ el

                print("### TERMINA PATRON ###")

            # Calculamos el error y evaluamos si se corta o no
            y = np.apply_along_axis(lambda p: self.evaluar(p).T, 1, x)
            #print(f'y: {y}, y.shape={y.shape}')
            #print(f'yd: {yd}')
            #print(f'(yd - y) / yd: {(yd - y) / yd}')
            err = np.average(np.abs((yd - y) / yd))

            print(f'Error de epoca: {err}')

            if err < umbral_err:
                break
