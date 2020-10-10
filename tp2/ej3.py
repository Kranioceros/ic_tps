import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import utils


def main():
    datos = np.genfromtxt("datos/circulo.csv", dtype=float, delimiter=',')
    (nro_patrones, dim_patrones) = datos.shape

    # Creamos grafica
    fig, ax = plt.subplots()
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)

    # Configuración del SOM
    (filas_som, cols_som) = (4, 4)
    nro_neuronas = filas_som * cols_som
    dist_entorno = 1    # Distancia de Hamming
    coef_apren = 0.2

    # Matriz con posiciones de las neuronas en el mapa
    M = []
    for i in range(filas_som):
        for j in range(cols_som):
            M.append([j, i])    # [x, y]

    M = np.array(M)

    # Matriz de segmentos (asocia indices de neuronas adyacentes)
    nro_segs = 2 * filas_som * cols_som - (filas_som + cols_som)
    S = np.zeros((nro_segs, 2), dtype=np.int32)
    k = 0
    # Segmentos horizontales
    for i in range(filas_som):
        for j in range(cols_som-1):
            S[k, 0] = i * cols_som + j
            S[k, 1] = i * cols_som + j + 1
            k += 1
    # Segmentos verticales
    for j in range(cols_som):
        for i in range(filas_som-1):
            S[k, 0] = i * cols_som + j
            S[k, 1] = (i+1) * cols_som + j
            k += 1

    print(f'S: {S}')

    # Mezclamos patrones
    idx_patrones = np.arange(nro_patrones)
    np.random.shuffle(idx_patrones)

    # Centroides correspondientes a las neuronas
    C = datos[idx_patrones][:nro_neuronas, :]

    for patron in datos:
        # Graficamos
        plt.cla()
        line_segs = LineCollection(C[S], colors='r', linestyle='dotted')
        ax.add_collection(line_segs)
        ax.scatter(datos[:, 0], datos[:, 1])
        ax.scatter(C[:, 0], C[:, 1], c='r', marker='D')
        plt.pause(0.001)

        # Desplazamiento del patron a cada centroide
        D = C - patron
        # Distancia euclidea ^ 2
        dists = D[:, 0]**2 + D[:, 1]**2

        # Buscamos el ganador y su coord. en el mapa
        idx_ganador = np.argmin(dists)
        coord_ganador = M[idx_ganador, :]
        xg = coord_ganador[0]
        yg = coord_ganador[1]

        # Calculamos mascara de entorno (descartamos los que superen la distancia
        # de hamming)
        mask = (np.abs(M[:, 0] - xg) + np.abs(M[:, 1] - yg)) <= dist_entorno

        # Ajustamos los centroides
        C[mask] = C[mask] - coef_apren * D[mask]

    # TODO
        # * Realizar varias epocas y realizar tres fases del algoritmo


if __name__ == "__main__":
    main()
