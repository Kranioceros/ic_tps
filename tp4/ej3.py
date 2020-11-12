import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import time

from ACO import Colonia, EstrategiaFeromona

def main():

    datos = np.genfromtxt("tp4/gtp4datos/gr17.csv", dtype=float, delimiter=',') 

    # La hormiga debe parar cuando se encuentre en el vertice 4
    def parada(p, _g):
        N = p.size
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
 
            return np.sum(d)+g[p[N-1],p[0]]

    #(evaporacion, q)
    pruebas = [(0.1, 0.1), (0.1, 1), (0.1, 10), (0.9, 0.1), (0.9, 1), (0.9, 10)]
    estrategias = [EstrategiaFeromona.Uniforme, EstrategiaFeromona.Local, EstrategiaFeromona.Global]
    a_b = [(1,0), (5,0), (1,3)]
    #(tiempo, distancia, iteraciones)
    resultados = []

    idx = 0
    for ab in a_b:
        for p in pruebas:
            for est in estrategias:
                colonia_kwargs = {
                    'n_hormigas' : 50,
                    'origen'     : 11,
                    'estrategia' : est,
                    'm_grafo'    : datos,
                    'f_parada'   : parada,
                    'f_costo'    : costo,
                    'sigma0'     : 0.001,
                    'alfa'       : ab[0],
                    'beta'       : ab[1],
                    'evaporacion': p[0],
                    'q'          : p[1],
                    'max_repet'  : 101,
                    'semilla'    : None,
                }

                colonia = Colonia(**colonia_kwargs)

                start = time.time()
                (camino, epocas, cst) = colonia.ejecutar(100, debug=False)
                tiempo = time.time() - start

                resultado_k = (tiempo, cst, epocas)

                resultados.append(resultado_k)

                print(f"DONE: {idx}")
                idx += 1

    estrategias_str = ['Uniforme', 'Local', 'Global']
    idx = 0
    for ab in a_b:
        print(f"---- Alpha = {ab[0]} | Bheta = {ab[1]} ----")
        for p in pruebas:
            print(f"---- Evaporacion = {p[0]} ----")
            for est in estrategias_str:
                print(f"q={p[1]} | {est} | Tiempo: {resultados[idx][0]} | Costo: {resultados[idx][1]} | Iteraciones: {resultados[idx][2]}")
                idx += 1
            print("---------")
        print('---------')


if __name__ == "__main__":
    main()