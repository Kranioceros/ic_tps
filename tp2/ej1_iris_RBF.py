import numpy as np
import matplotlib.pyplot as plt
import math
import utils
from NN import NN


def main():
    neuronasRadiales = 20
    nnMultiCapa = NN([neuronasRadiales,3], learning_rate=.1)

    datos = np.genfromtxt("datos/irisbin.csv", dtype=float, delimiter=',')

    particion = utils.particionar(datos, 1, .8, random=True)

    #Patrones y etiquetas de entrenamiento
    m_inputs_trn = datos[particion[0][0],:-3]
    v_labels_trn = datos[particion[0][0], -3:]

    #Patrones y etiquetas de testeo
    m_inputs_tst = datos[particion[0][1],:-3]
    v_labels_tst = datos[particion[0][1], -3:]

    idx = np.arange(m_inputs_trn.shape[0])
    dimension = m_inputs_trn.shape[1]

    #Matriz de medias (tantas filas como neuronas radiales, y tantas columnas como dimension del problema)
    medias = np.zeros((neuronasRadiales, dimension))
    np.random.shuffle(idx)
    for i in range(neuronasRadiales):
        medias[i,:] = m_inputs_trn[idx[i]]


    asignaciones_ant = [1]
    asignaciones = []

    while(asignaciones != asignaciones_ant):
        asignaciones_ant = asignaciones
        asignaciones = []
        for patron in m_inputs_trn:
            distanciasPatronMedias = []
            for media in medias:
                distanciasPatronMedias.append(np.sqrt(np.sum((media-patron)**2))) #Raíz cuadrada de la suma de los cuadrados de las diferencias
            min_idx = np.argmin(distanciasPatronMedias)
            asignaciones.append(min_idx) #este vector me dice para cada patron, que media le corresponde
        
        for idx_media in range(medias.shape[0]):
            patronesMediaI = []

            for i in range(len(asignaciones)):
                if(asignaciones[i] == idx_media):
                    patronesMediaI.append(i)

            m_inputs_media = m_inputs_trn[patronesMediaI] 

            x1_prom = 0
            x2_prom = 0
            x3_prom = 0
            x4_prom = 0

            if(len(m_inputs_media[:,0])!=0):
                x1_prom = np.mean(m_inputs_media[:,0])
            if(len(m_inputs_media[:,1])!=0):
                x2_prom = np.mean(m_inputs_media[:,1])
            if(len(m_inputs_media[:,2])!=0):
                x3_prom = np.mean(m_inputs_media[:,2])
            if(len(m_inputs_media[:,3])!=0):
                x4_prom = np.mean(m_inputs_media[:,3])
      
            medias[idx_media,:] = [x1_prom, x2_prom, x3_prom, x4_prom]

        #print(asignaciones)

    #Tenemos todas las medias, hay que calcular las salidas de las gaussianas. 
    m_inputs_perceptron = np.zeros((m_inputs_trn.shape[0],neuronasRadiales))
    for idx_p,p in enumerate(m_inputs_trn):
        for idx_m,m in enumerate(medias):
            m_inputs_perceptron[idx_p,idx_m] = utils.gaussiana(p,m,1)

    epocas_convergencia_iteracion = nnMultiCapa.Train(m_inputs_perceptron,v_labels_trn, max_epochs=300, tol_error=.15)

    #TERMINA ENTRENAMIENTO
    #------------------------------------------------------------------------------------------------------------------------
    #TESTEO

    #Feed fodward capa radial
    m_inputs_test = np.ones((m_inputs_tst.shape[0],neuronasRadiales))
    for idx_p_test,p_test in enumerate(m_inputs_tst):
        for idx_m,m in enumerate(medias):
            m_inputs_test[idx_p_test,idx_m] = utils.gaussiana(p_test,m,1)

    #Feed fodward perceptrones simples
    resultados = np.zeros(v_labels_tst.shape)
    errores = []
    v_clase1 = []
    v_clase2 = []
    v_clase3 = []
    for idx_p, p_test in enumerate(m_inputs_test):
        #Salida de la red con WinnerTakesAll
        output = utils.WinnerTakesAll(nnMultiCapa.Test(p_test)[0][:])
        #Error de la salida (error tiene 1s y 0s, si son todos 0 no hay error)
        error = np.abs(output-v_labels_tst[idx_p])
        if(sum(error)!=0):
            errores.append(1)
        else:
            errores.append(0)

        #Cómo fue clasificado el patrón
        clase_output = np.argmax(output)
        if(clase_output==0):
            v_clase1.append(idx_p)
        elif(clase_output == 1):
            v_clase2.append(idx_p)
        else:
            v_clase3.append(idx_p)

    print(f"MEDIA ERROR: {np.mean(errores)}")
    print(f"STD ERROR: {np.std(errores)}")

    #Grafico de petalos
    plt.scatter(m_inputs_tst[v_clase1,0], m_inputs_tst[v_clase1,1], color=(1,0,0), label="Virginica")
    plt.scatter(m_inputs_tst[v_clase2,0], m_inputs_tst[v_clase2,1], color=(0,0,1), label="Versicolor")
    plt.scatter(m_inputs_tst[v_clase3,0], m_inputs_tst[v_clase3,1], color=(0,1,0), label="Setosa")
    
    plot_circles = False
    if(plot_circles):
        for m in medias:
            circle = plt.Circle((m[0], m[1]), 1, fill=False, edgecolor=(0,0,0), linewidth='1')
            plt.gca().add_patch(circle)
    else:
        plt.scatter(medias[:,0], medias[:,1], color=(0,0,0), label="Centroides")

    plt.xlabel("Ancho (cm)")
    plt.ylabel("Alto (cm)")
    plt.title("Pétalos RBF")
    plt.legend(loc="lower right", frameon=False)
    plt.show() 


    #Grafico de sepalos
    plt.scatter(m_inputs_tst[v_clase1,2], m_inputs_tst[v_clase1,3], color=(1,0,0), label="Virginica")
    plt.scatter(m_inputs_tst[v_clase2,2], m_inputs_tst[v_clase2,3], color=(0,0,1), label="Versicolor")
    plt.scatter(m_inputs_tst[v_clase3,2], m_inputs_tst[v_clase3,3], color=(0,1,0), label="Setosa")
    
    if(plot_circles):
        for m in medias:
            circle = plt.Circle((m[2], m[3]), 1, fill=False, edgecolor=(0,0,0), linewidth='1')
            plt.gca().add_patch(circle)
    else:
        plt.scatter(medias[:,2], medias[:,3], color=(0,0,0), label="Centroides")

    plt.xlabel("Ancho (cm)")
    plt.ylabel("Alto (cm)")
    plt.legend(loc="lower right", frameon=False)
    plt.title("Sépalos RBF")
    plt.show() 



if __name__ == "__main__":
    main()