import numpy as np
from matplotlib import pyplot as plt

from ACO import Colonia, EstrategiaFeromona

prueba = 1

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

    m_grafo = np.array([[0, 5, 1, 50],
                        [5, 0, 50, 2],
                        [1, 50, 0, 10],
                        [50, 2, 10, 0]])

    datos = np.genfromtxt("gtp4datos/gr17.csv", dtype=float, delimiter=',') 

    # La hormiga debe parar cuando se encuentre en el vertice 4
    def parada(p, _g):
        N = p.size
        if(prueba==0):
            return p[N-1] == 3
        else:
            return N == datos.shape[0]

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
            if(prueba==1):
                d += g[p[N-1], p[0]]
            
            return np.sum(d)

    # Colonia(semilla=None, n_hormigas, origen, estrategia, m_grafo, f_parada, f_costo, sigma0, alfa, evaporacion):
    
    if(prueba==0):
        colonia_kwargs = {
            'n_hormigas' : 100,
            'origen'     : 0,
            'estrategia' : EstrategiaFeromona.Global,
            'm_grafo'    : m_grafo,
            'f_parada'   : parada,
            'f_costo'    : costo,
            'sigma0'     : 0.001,
            'alfa'       : 2,
            'beta'       : 0,
            'evaporacion': 0.1,
            'q'          : 1,
            'max_repet'  : 50,
            'semilla'    : None,
        }
    else:
        colonia_kwargs = {
            'n_hormigas' : 100,
            'origen'     : 11,
            'estrategia' : EstrategiaFeromona.Global,
            'm_grafo'    : datos,
            'f_parada'   : parada,
            'f_costo'    : costo,
            'sigma0'     : 0.001,
            'alfa'       : 2,
            'beta'       : 0,
            'evaporacion': 0.1,
            'q'          : 1,
            'max_repet'  : 50,
            'semilla'    : None,
        }

    colonia = Colonia(**colonia_kwargs)

    (res, _epocas, cst) = colonia.ejecutar(100, debug=False)
    print(f"Res: {res} | Epocas: {_epocas} | Costo: {cst}")

if __name__ == "__main__":
    main()