import numpy as np
import matplotlib.pyplot as plt
from neurona import Neurona
from utils import extender


def main():
    # Leemos archivo de entrenamiento
    datos = np.genfromtxt("tp1/icgtp1datos/OR_trn.csv",
                          dtype=float, delimiter=',')
    # Agregamos una entrada constante -1 para el sesgo
    datos_ext = extender(datos)

    # Creamos una neurona con tres entradas
    neurona = Neurona(dim=3)
    neurona.aleatorizar(-0.5, 0.5)

    # Entrenamos
    (_err, _epocas, pesos) = neurona.entrenar(
        datos_ext, umbral_err=0.05, coef_apren=0.1, guardar_pesos=True)

    # Visualizamos
    print("Visualizando entrenamiento con funcion OR...")
    visualizar(pesos, datos_ext, max_pasos=50)

    # Leemos otro archivo de entrenamiento
    datos = np.genfromtxt("tp1/icgtp1datos/XOR_trn.csv",
                          dtype=float, delimiter=',')
    # Agregamos una entrada constante -1 para el sesgo
    datos_ext = extender(datos)

    # Entrenamos
    neurona.aleatorizar(-0.5, 0.5)

    (_err, _epocas, pesos) = neurona.entrenar(
        datos_ext, umbral_err=0.05, coef_apren=0.1, guardar_pesos=True)

    # Visualizamos
    print("Visualizando entrenamiento con funcion XOR...")
    visualizar(pesos, datos_ext, max_pasos=50)

# Toma pesos de una neurona datos de entrenamiento y visualiza de forma animada
# como ajusta sus parametros
# `pesos` es una matriz con los pesos de las neuronas en cada paso de entrenamiento


def visualizar(pesos, datos_entr, max_pasos=200, t=0.1):
    (_fil, col) = datos_entr.shape

    if len(pesos[0]) != 3 or col != 4:
        print("Error de dimensiones en visualizar")

    n = min(len(pesos), len(datos_entr), max_pasos)
    for i in range(n):
        w = pesos[i]
        x1 = datos_entr[i, 1]
        x2 = datos_entr[i, 2]
        if datos_entr[i, 3] == 1:
            plt.scatter(x1, x2, c='b')
        else:
            plt.scatter(x1, x2, c='r')
        plt.axline((0, w[0]/w[2]), slope=-w[1]/w[2])
        plt.axis([-1.1, 1.1, -1.1, 1.1])
        plt.pause(t)
        plt.cla()


# Ejecutar la funcion main solo si el script es ejecutado directamente
if __name__ == "__main__":
    main()
