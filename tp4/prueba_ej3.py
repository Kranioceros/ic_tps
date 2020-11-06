import numpy as np
from matplotlib import pyplot as plt

from ACO import Colonia, EstrategiaFeromona

def main():
    # Para verificar el funcionamiento correcto del algoritmo, es mas facil
    # probar primero con un experimento de puente binario
    #
    #       --- v2 ---
    #   v1 -|        |- v4
    #       --- v3 ---
    #
    # Origen:   v1
    # Objetivo: v4
    # Costo:
    #   v1 a v2: 5
    #   v1 a v3: 3
    #   v2 a v4: 3
    #   v3 a v4: 6
    #   otros: 50
    #
    # La solucion optima es (0, 1, 3)
    # La segunda mejor es (0, 2, 3)

    m_grafo = np.array([[0, 5, 3, 50],
                        [5, 0, 50, 3],
                        [3, 50, 0, 6],
                        [50, 3, 6, 0]])

    # La hormiga debe parar cuando se encuentre en el vertice 4
    def parada(p, _g):
        N = p.size
        return p[N-1] == 3

    # El costo del camino es la suma de las distancias de las aristas
    def costo(p, g):
        # Numero de vertices en el camino
        N = p.size
        if N == 2:
            return g[p[0], p[1]]
        else:
            # Indices para obtener una matriz de 2x(N-1) con las posiciones de los
            # extremos de cada arista.
            idx =  np.arange(-1, 1)[:, None] + (np.arange(N-1) + 1)[None, :]
            idx = p[idx]
            # Vector con las distancias de cada arista
            d = g[idx[:, 0], idx[:, 1]]
            return np.sum(d)

    # Colonia(semilla=None, n_hormigas, origen, estrategia, m_grafo, f_parada, f_costo, sigma0, alfa, evaporacion):
    colonia_kwargs = {
        'n_hormigas' : 20,
        'origen' : 0,
        'estrategia' : EstrategiaFeromona.Local,
        'm_grafo'    : m_grafo,
        'f_parada'   : parada,
        'f_costo'    : costo,
        'sigma0'     : 0.2,
        'alfa'       : 0.8,
        # Falta beta aca
        'evaporacion': 0.3,
        'q'          : 0.1,
        'semilla'    : 324098475,
    }

    colonia = Colonia(**colonia_kwargs)

    (res, _epocas) = colonia.ejecutar(100, debug=True)

if __name__ == "__main__":
    main()