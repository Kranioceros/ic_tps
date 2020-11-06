import numpy as np
from enum import Enum

class Colonia:
    # Una colonia se define por:
    #   * n_hormigas: Nro. de hormigas
    #   * origen: Vertice origen del grafo
    #   * m_grafo: Matriz cuadrada con costo entre vertice y vertice. El grafo
    #              debe ser totalmente conexo
    #   * estrategia: Estrategia para calcular el delta de feromonas (objeto de tipo `EstrategiaFeromona`)
    #   * f_parada: Recibe el camino recorrido hasta el momento y el grafo.
    #               Devuelve un booleano especificando si se encontro el objetivo o no.
    #   * f_costo:  Recibe el camino recorrido hasta el momento y el grafo.
    #               Devuelve el costo del camino recorrido.
    #   * sigma0: Valor maximo que pueden tomar las feromonas
    #   * alfa: Valor al cual se eleva el rastro de feromonas
    #   * q: Valor base de feromona a agregar en cada ciclo
    #   * semilla: `None` si se inicializa al azar, sino un numero
    #
    # Un camino es un ndarray unidimensional, con tantos elementos como vertices
    # pase el camino
    def __init__(self, n_hormigas, origen, estrategia, m_grafo,
        f_parada, f_costo, sigma0, alfa, evaporacion, q, semilla=None):
        self.n_hormigas = n_hormigas
        self.origen = origen
        self.estrategia = estrategia
        self.m_grafo = m_grafo
        self.f_parada = f_parada
        self.f_costo = f_costo
        self.alfa = alfa
        self.evaporacion = evaporacion
        self.q = q

        # Inicializamos generador de numeros
        self.rng = np.random.default_rng(semilla)

        # Vector con feromonas para cada arista
        self.m_feromonas = sigma0 * self.rng.random(m_grafo.shape)
        
        # Matriz con caminos de todas las hormigas. El camino mas largo posible
        # es de tamaño `N`. Se inicializa en 0.
        self.N = m_grafo.shape[0]
        self.m_caminos = np.zeros((n_hormigas, self.N))

        # Vector con los costos actuales de los caminos recorridos por cada
        # hormiga.
        self.v_costos = np.zeros(n_hormigas)

    # Ejecuta el algoritmo
    # max_epocas: define el numero de ciclos de busqueda a ejecutar como maximo
    #   si los caminos de las hormigas no convergen
    # debug: Si es `True`, se imprimen mensajes de Debug. Por defecto esta en `False`
    # Devuelve el camino al cual convergen las hormigas y el numero de ciclos
    # de busqueda. Si no convergen, devuelve la matriz de caminos.
    def ejecutar(self, max_epocas, debug=False):
        dbg = print if debug else lambda x: None

        for epoca in range(max_epocas):
            print(f'####################')
            print(f'#### Epoca {epoca:4d} ####')
            print(f'####################')

            # Matriz 3D que almacena para cada arista cuales hormigas
            # la visitaron
            m_visitas = np.zeros((self.N, self.N, self.n_hormigas), dtype=np.int)

            for h in range(self.n_hormigas):
                #dbg(f'~~ Hormiga {h}~~')
                # Reiniciamos la posicion de la hormiga
                vert_actual = self.origen
                # Inicializamos camino
                largo_camino = 1
                camino = np.ones(self.N, dtype=np.int) * (-1)
                camino[0] = vert_actual
                # Inicializamos vector de vertices no visitados
                no_visitados = np.ones(self.N, dtype=bool)
                no_visitados[vert_actual] = False

                # Mientras la hormiga no haya alcanzado su objetivo
                while(not self.f_parada(camino[:largo_camino], self.m_grafo)):
                    #dbg(f'Vertice actual: {vert_actual}')

                    # Buscamos los vertices adyacentes no visitados
                    ady = np.flatnonzero(no_visitados)
                    #dbg(f'Adyacentes: {ady}')

                    # Buscamos el rastro de feromonas para todas las aristas posibles
                    ferom_arista = self.m_feromonas[vert_actual, ady]
                    #dbg(f'Feromonas en las aristas: {ferom_arista}')

                    # Calculamos la funcion de deseo para todas las aristas. Es la
                    # inversa de la distancia. Funciona siempre porque descartamos
                    # los ceros anteriormente
                    deseo_arista = 1 / (self.m_grafo[vert_actual, ady])
                    #dbg(f'Deseo de las aristas: {deseo_arista}')

                    # Calculamos las probabilidades de todas las aristas disponibles
                    # FALTA PARAMETRO BETA
                    prob_arista = ((ferom_arista ** self.alfa) * deseo_arista /
                                np.dot(ferom_arista ** self.alfa, deseo_arista))
                    #dbg(f'Probabilidad de las aristas: {prob_arista}')

                    # Seleccionamos un vertice en base a las probabilidades de su
                    # arista.
                    nuevo_vert = self.rng.choice(ady, p=prob_arista)
                    #dbg(f'Vertice seleccionado: {nuevo_vert}')

                    # Actualizamos la posicion de la hormiga y el camino
                    vert_actual = nuevo_vert
                    camino[largo_camino] = nuevo_vert
                    largo_camino += 1

                    # Actualizamos no visitados
                    no_visitados[nuevo_vert] = False
                # Guardamos el camino
                self.m_caminos[h] = camino
                dbg(f'Objetivo alcanzado con p: {camino[:largo_camino]}')
                # Guardamos el costo del camino
                self.v_costos[h] = self.f_costo(camino[:largo_camino],
                                                self.m_grafo)
                dbg(f'Costo del camino: {self.v_costos[h]}')
                # Marcamos las aristas visitadas
                idx = aristas_camino(camino[:largo_camino], self.m_grafo)
                m_visitas[idx[0], idx[1], h] = 1

            # Si los caminos son todos iguales, encontramos la solucion
            iguales = True
            for i in range(self.n_hormigas - 1):
                if np.array_equal(self.m_caminos[i], self.m_caminos[i+1]):
                    continue
                else:
                    iguales = False
                    break
            if iguales:
                return (self.m_caminos[0], epoca+1)

            # Evaporamos las feromonas
            self.m_feromonas = (1 - self.evaporacion) * self.m_feromonas

            # Aplicamos alguna de las estrategias para colocar feromonas

            if self.estrategia == EstrategiaFeromona.Uniforme:
                self.m_feromonas += self.q * np.sum(m_visitas, 2)
            elif self.estrategia == EstrategiaFeromona.Local:
                delta = np.sum(m_visitas, 2) / self.m_grafo
                delta = np.ma.array(delta, mask=np.isnan(delta))
                self.m_feromonas += self.q * delta
            else:
                # POR HACER
                pass
            
        return (self.m_caminos, max_epocas)
        
def aristas_camino(p, g):
    N = p.size
    if N == 2:
        return p[:, None]
    else:
        idx = np.arange(-1, 1)[:, None] + (np.arange(N-1) + 1)[None, :]
        return p[idx]


EstrategiaFeromona = Enum('EstrategiaFeromona', 'Global Uniforme Local')
