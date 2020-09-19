from multicapa import RedMulticapa
import numpy as np
import matplotlib.pyplot as plt
from utils import configurar_grilla, sig, dsig, signo, const


def main():
    # Creamos figura final con subplots
    _fig, axs = plt.subplots(2, 2)

    # Creamos una red neuronal que resuelva el problema XOR
    # 2 entradas (x1 y x2), 2 neuronas en la capa de entrada (n00 y n01), 1 neurona en la capa de salida (n10)
    nn = RedMulticapa([2, 2, 1], activ=signo)

    # Asignamos pesos a las capas
    nn.ws[0] = np.array([[-1, 1, 1], [1, 1, 1]])
    nn.ws[1] = np.array([[1, 1, -1]])

    # Graficamos pesos de las tres neuronas
    graficar_pesos(nn, axs[0, 0], 'Pesos de red construida')

    # Cargamos los datos de testeo de xor
    xor_tst = np.genfromtxt("tp1/icgtp1datos/XOR_tst.csv",
                            dtype=float, delimiter=',')

    x = xor_tst[:, :-1]       # Entrada, un patron por fila

    # Clasificamos
    graficar_clasificacion(
        nn, x, axs[1, 0], 'Clasificacion de red construida')

    # Cargamos los datos de entrenamiento de xor
    xor_trn = np.genfromtxt("tp1/icgtp1datos/XOR_trn.csv",
                            dtype=float, delimiter=',')

    # Creamos una red identica pero la entrenamos de cero
    nn = RedMulticapa([2, 2, 1], interv_rand=(-0.5, 0.5))

    nn.entrenar(xor_trn, max_epocas=5, coef_apren=0.2)

    # Graficamos pesos de las tres neuronas
    graficar_pesos(nn, axs[0, 1], 'Pesos de red entrenada')

    # Clasificamos
    graficar_clasificacion(
        nn, x, axs[1, 1], 'Clasificacion de red entrenada')

    plt.show()

# Toma la red neuronal, un Axes y grafica las rectas que definen los pesos


def graficar_pesos(nn, ax, title=None):
    # Pesos neuronas de la capa 0
    n00 = nn.ws[0][0, :]
    n01 = nn.ws[0][1, :]
    # Pesos neurona de la capa 1
    n10 = nn.ws[1][0, :]

    # Graficamos rectas de las neuronas
    ax.axline((0, n00[0]/n00[2]), slope=-n00[1] / n00[2], c='r', label="n00")
    ax.axline((0, n01[0]/n01[2]), slope=-n01[1] / n01[2], c='g', label="n01")
    ax.axline((0, n10[0]/n10[2]), slope=-n10[1] / n10[2], c='b', label="n10")

    # Configuramos axes
    configurar_grilla(ax)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()
    if title is not None:
        ax.set_title(title)

# Toma la red neuronal, los patrones, un Axes y grafica la clasificacion
# que la red hace de los patrones


def graficar_clasificacion(nn, x, ax, title=None):
    # Evaluamos la red para cada patron de prueba
    #(fils, _cols) = x.shape
    #y = np.zeros(fils)
    # for i in range(fils):
    #y[i] = nn.evaluar(x[i])

    y = np.apply_along_axis(nn.evaluar, 1, x).flatten()

    # Graficamos patrones
    idx_true = y > 0
    idx_false = y < 0
    ax.scatter(x[idx_true, 0], x[idx_true, 1], c='b', label='true')
    ax.scatter(x[idx_false, 0], x[idx_false, 1], c='r', label='false')

    # Configuramos axes
    configurar_grilla(ax)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()

    if title is not None:
        ax.set_title(title)


# Se corre main si el script se ejecuta directamente
if __name__ == "__main__":
    main()
