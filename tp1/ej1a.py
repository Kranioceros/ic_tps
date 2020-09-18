import numpy as np
import matplotlib.pyplot as plt
from neurona import Neurona
from utils import extender, configurar_grilla, visualizar2D


def main():
    # Leemos archivo de entrenamiento
    or_trn = np.genfromtxt("tp1/icgtp1datos/OR_trn.csv",
                           dtype=float, delimiter=',')
    # Agregamos una entrada constante -1 para el sesgo
    or_trn_ext = extender(or_trn)

    # Creamos una neurona con tres entradas
    neurona = Neurona(dim=3)
    neurona.aleatorizar(-0.5, 0.5)

    # Entrenamos
    (_err, _epocas, pesos) = neurona.entrenar(
        or_trn_ext, umbral_err=0.05, coef_apren=0.1, guardar_pesos=True)

    # Visualizamos de forma animada
    print("Visualizando entrenamiento con funcion OR...")
    visualizar2D(pesos, or_trn, max_pasos=50,
                 titulo="Entrenamiento con OR", labels=('x1', 'x2'))

    # Leemos archivo de prueba
    or_tst = np.genfromtxt("tp1/icgtp1datos/OR_tst.csv",
                           dtype=float, delimiter=',')

    # Agregamos una entrada constante -1 para el sesgo
    or_tst_ext = extender(or_tst)

    # Evaluamos para los patrones de prueba
    y = np.apply_along_axis(neurona.evaluar, 1, or_tst_ext[:, :-1])

    # Configuramos nueva grafica
    ax = plt.subplot()
    configurar_grilla(ax)
    ax.set_title('Resultados del clasificador con patrones de prueba')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    # Graficamos patrones
    idx_true = y > 0
    idx_false = y < 0
    x1 = or_tst[:, 0]
    x2 = or_tst[:, 1]
    plt.scatter(x1[idx_true],  x2[idx_true], c='b', label='True')
    plt.scatter(x1[idx_false], x2[idx_false], c='r', label='False')

    # Graficamos recta de los pesos sinapticos
    neurona.graficar2(ax)

    # Generamos leyenda y mostramos
    ax.legend()
    plt.show()


# Ejecutar la funcion main solo si el script es ejecutado directamente
if __name__ == "__main__":
    main()
