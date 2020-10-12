import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import utils


def main():
    datos_completos = np.genfromtxt("datos/clouds.csv", dtype=float, delimiter=',')
   

    #Particionado
    particiones = utils.particionar(datos_completos, 1, .8, random=True)

    #Datos y etiquetas de entrenamiento
    datos_trn = datos_completos[particiones[0][0],:-1]
    etiquetas_trn = datos_completos[particiones[0][0],-1]

    (nro_patrones, dim_patrones) = datos_trn.shape

    #Datos y etiquetas de testeo
    datos_tst = datos_completos[particiones[0][1],:-1]
    etiquetas_tst = datos_completos[particiones[0][1],-1]

    # Creamos grafica
    fig, ax = plt.subplots()
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)

    # Configuración del SOM
    (filas_som, cols_som) = (10, 10)
    nro_neuronas = filas_som * cols_som
    dist_entorno = 5    # Distancia de Hamming
    coef_apren = 0.2
    max_epocas = 1000
    plt_dinamico = False

    # Matriz con posiciones de las neuronas en el mapa
    M = []
    for i in range(filas_som):
        for j in range(cols_som):
            M.append([j, i])    # [x, y]

    M = np.array(M)

    #Matriz de etiquetas de neuronas
    etiquetas_neuronas = np.zeros((nro_neuronas, 2))

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

    #print(f'S: {S}')

    # Mezclamos patrones
    idx_patrones = np.arange(nro_patrones)
    np.random.shuffle(idx_patrones)

    # Centroides correspondientes a las neuronas
    C = datos_trn[idx_patrones][:nro_neuronas, :]

    for epoca in range(max_epocas):
        for idx_patron, patron in enumerate(datos_trn):
            # Graficamos
            if(plt_dinamico==True):
                plt.cla()
                line_segs = LineCollection(C[S], colors='r', linestyle='dotted')
                ax.add_collection(line_segs)
                ax.scatter(datos_trn[:, 0], datos_trn[:, 1])
                ax.scatter(C[:, 0], C[:, 1], c='r', marker='D')
                plt.pause(0.001)
            elif(epoca==999 and idx_patron==nro_patrones-1):
                line_segs = LineCollection(C[S], colors='r', linestyle='dotted')
                ax.add_collection(line_segs)
                ax.scatter(datos_trn[:, 0], datos_trn[:, 1])
                ax.scatter(C[:, 0], C[:, 1], c='r', marker='D')
                plt.show()

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
            mask = (np.abs(M[:, 0] - xg) + np.abs(M[:, 1] - yg)) <= dist_entorno*utils.adaptarParametro(epoca, 200, 500)

            # Ajustamos los centroides
            C[mask] = C[mask] - coef_apren*utils.adaptarParametro(epoca, 200, 500) * D[mask]
        print(f"epoca: {epoca}")

    plt.close()

    #Establecer etiquetas de cada neurona
    for idx_patron, patron in enumerate(datos_trn):
        # Desplazamiento del patron a cada centroide
        D = C - patron
        # Distancia euclidea ^ 2
        dists = D[:, 0]**2 + D[:, 1]**2

        # Buscamos el ganador y su coord. en el mapa
        idx_ganador = np.argmin(dists)
        
        if(etiquetas_trn[idx_patron] == 1):
            etiquetas_neuronas[idx_ganador][1] += 1
        else:
            etiquetas_neuronas[idx_ganador][0] += 1
        
    #----------------------------------------------------------------
    
    #Test
    v_error = []
    for idx_patron_tst, patron_tst in enumerate(datos_tst):

        # Desplazamiento del patron a cada centroide
        D_tst = C - patron_tst
         # Distancia euclidea ^ 2
        dists = D_tst[:, 0]**2 + D_tst[:, 1]**2

        # Buscamos el ganador y su coord. en el mapa
        idx_ganador = np.argmin(dists)

        if(etiquetas_tst[idx_patron_tst] == np.argmax(etiquetas_neuronas[idx_ganador][:])):
            v_error.append(0)
        else:
            v_error.append(1)

    print(f"Media Error: {np.mean(v_error)}")

if __name__ == "__main__":
    main()
