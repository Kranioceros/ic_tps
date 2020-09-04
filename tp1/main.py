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
    neurona.entrenar(datos_ext, max_epocas=1)

    # Graficamos
    plt.scatter(datos[:, 0], datos[:, 1])
    w = neurona.w
    plt.axline((0, w[0]/w[2]), slope=-w[1]/w[2])
    plt.show()

    # Leemos archivo de prueba
    datos = np.genfromtxt("tp1/icgtp1datos/OR_tst.csv",
                          dtype=float, delimiter=',')
    datos_ext = extender(datos[:, :-1])

    # Evaluamos la salida
    y = neurona.evaluar(datos_ext)

    # Indices para patrones verdaderos o falsos
    idx_v = y == 1
    idx_f = y == -1

    plt.scatter(datos[idx_v, 0], datos[idx_v, 1], c="g")
    plt.scatter(datos[idx_f, 0], datos[idx_f, 1], c="r")
    plt.show()


# Ejecutar la funcion main solo si el script es ejecutado directamente
if __name__ == "__main__":
    main()
