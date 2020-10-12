import numpy as np
import matplotlib.pyplot as plt
import math
import utils
from NN import NN

def main():
    datos = np.genfromtxt("datos/merval.csv", dtype=float, delimiter=',')

    (datos, max_dato) = utils.NormalizarDatos(datos)

    cant_patrones = datos.shape[0]-5

    m_datos = np.zeros((cant_patrones,6))
    k = 0
    for i in range(cant_patrones):
        m_datos[i,:] = datos[k:k+6]
        k+=1

    particion = utils.particionar(m_datos, 1, .8, random=False)

    #Patrones y etiquetas de entrenamiento
    m_inputs_trn = m_datos[particion[0][0],:-1]
    v_labels_trn = m_datos[particion[0][0], -1:]

    #Patrones y etiquetas de testeo
    m_inputs_tst = m_datos[particion[0][1],:-1]
    v_labels_tst = m_datos[particion[0][1], -1:]

    idx = np.arange(m_inputs_trn.shape[0])
    dimension = m_inputs_trn.shape[1]

    neuronasRadiales = 5
    nnMultiCapa = NN([neuronasRadiales,1], learning_rate=.1, activation=utils.identidad, dactivation=utils.identidad)

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
            x5_prom = 0
            if(len(m_inputs_media[:,0])!=0):
                x1_prom = np.mean(m_inputs_media[:,0])
            if(len(m_inputs_media[:,1])!=0):
                x2_prom = np.mean(m_inputs_media[:,1])
            if(len(m_inputs_media[:,2])!=0):
                x3_prom = np.mean(m_inputs_media[:,2])
            if(len(m_inputs_media[:,3])!=0):
                x4_prom = np.mean(m_inputs_media[:,3])
            if(len(m_inputs_media[:,4])!=0):
                x5_prom = np.mean(m_inputs_media[:,4])
      
            medias[idx_media,:] = [x1_prom, x2_prom, x3_prom, x4_prom, x5_prom]

        #print(asignaciones)

    #Tenemos todas las medias, hay que calcular las salidas de las gaussianas. 
    m_inputs_perceptron = np.zeros((m_inputs_trn.shape[0],neuronasRadiales))
    for idx_p,p in enumerate(m_inputs_trn):
        for idx_m,m in enumerate(medias):
            m_inputs_perceptron[idx_p,idx_m] = utils.gaussiana(p,m,1)

    epocas_convergencia_iteracion = nnMultiCapa.Train(m_inputs_perceptron,v_labels_trn, max_epochs=300, tol_error=.25)

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
    eficacias = []
    for idx_p, p_test in enumerate(m_inputs_test):
        resultados[idx_p]=nnMultiCapa.Test(p_test)[0,0]
        eficacia = resultados[idx_p]/v_labels_tst[idx_p]
        eficacias.append(eficacia)
        print(f"Esperado: {v_labels_tst[idx_p]*max_dato} | Obtenido: {resultados[idx_p]*max_dato} | Eficacia: {eficacia}")

    #El mejor caso para la eficacia es que sea 1
    #Lo esperado es que la media se encuentre cerca de 1
    #Tener en cuenta que puede ser mayor a 1 (si la salida de la red es mas grande que la etiqueta) o menor a 1 (si la salida de la red es menor que la etiqueta)
    print(f"Media Eficacias: {np.mean(eficacias)}")

    plt.scatter(np.arange(0,len(m_inputs_tst)), resultados[:,0]*max_dato, color=(1,0,0), label="Prediccion")
    plt.scatter(np.arange(0,len(m_inputs_tst)), v_labels_tst[:,0]*max_dato, color=(0,1,0), label="Real")
    
    plt.xlabel("Día")
    plt.ylabel("Índice MERVAL")
    plt.legend(loc="lower right", frameon=False)
    plt.title("Prediccion MERVAL RBF")
    plt.show()

if __name__ == "__main__":
    main()