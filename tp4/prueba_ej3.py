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

    datos = np.genfromtxt("tp4/gtp4datos/gr17.csv", dtype=float, delimiter=',') 


    # La hormiga debe parar cuando se encuentre en el vertice 4
    def parada(p, _g):
        N = p.size
        return N == datos.shape[0]
        #return p[N-1] == 3

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
            d = g[idx[0, :], idx[1, :]]
 
            return np.sum(d)+g[p[N-1],p[0]]

    # Colonia(semilla=None, n_hormigas, origen, estrategia, m_grafo, f_parada, f_costo, sigma0, alfa, evaporacion):
    colonia_kwargs = {
        'n_hormigas' : 100,
        'origen'     : 11,
        'estrategia' : EstrategiaFeromona.Global,
        'm_grafo'    : datos,
        'f_parada'   : parada,
        'f_costo'    : costo,
        'sigma0'     : 0.01,
        'alfa'       : 1.6,
        'beta'       : 0,
        'evaporacion': 0.1,
        'q'          : 0.01,
        'semilla'    : None,
    }

    colonia = Colonia(**colonia_kwargs)

    (res, _epocas) = colonia.ejecutar(1000, debug=False)
    print(res)
    print(colonia.m_feromonas)
    v = np.array([11, 13,  5, 14,  1,  4, 16,  7, 15,  0, 12,  6,  3,  2,  8, 10,  9])
    print("COSTOS:" ,costo(v, datos))
if __name__ == "__main__":
    main()