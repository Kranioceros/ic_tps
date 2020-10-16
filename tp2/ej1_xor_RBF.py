import numpy as np
import matplotlib.pyplot as plt
import math
import utils
from NN import NN

#Primera parte de la RBF

#inicializamos las medias de las gausianas
#para inicializar las medias, o formo k grupos aleatorios, o agarro k patrones y los usio como medias iniciales
#1 por cada patron del archivo, calculamos la distancia a cada media de las neuronas.
#2 asigno cada patron al grupo o neurona que tiene la media mas cercana al mismo
#3 con los patrones asignados a cada media, recalculo las medias existentes.
#4 si no hay mas reasignaciones, corto
#5 vuelvo al paso 1


def main():
    neuronasRadiales = 2
    nnMultiCapa = NN([neuronasRadiales,1], learning_rate=.3)

    datos = np.genfromtxt("datos/XOR_trn.csv", dtype=float, delimiter=',')
    datosTest = np.genfromtxt("datos/XOR_tst.csv", dtype=float, delimiter=',')

    sigma = .1

    #Matriz de patrones sin etiquetas
    m_inputs = datos[:,:-1]
    #Vector de etiquetas de los patrones
    v_labels = datos[:, -1:]

    idx = np.arange(m_inputs.shape[0])
    dimension = m_inputs.shape[1]

    medias = np.zeros((neuronasRadiales, dimension))
    
    np.random.shuffle(idx)
    #for i in range(neuronasRadiales):
    #    medias[i,:] = m_inputs[idx[i]]

    #Cantidad de patrones en cada conjunto
    porConjunto = int(m_inputs.shape[0]/neuronasRadiales)
    for i in range(neuronasRadiales):
        start = i*porConjunto
        end = start+porConjunto
        medias[i,:] = np.mean(m_inputs[idx[start:end]])

    asignaciones_ant = [1]
    asignaciones = []

    while(asignaciones != asignaciones_ant):
        asignaciones_ant = asignaciones
        asignaciones = []
        for patron in m_inputs:
            distanciasPatronMedias = []
            for media in medias:
                #distanciasPatronMedias.append([math.sqrt((media[0]-patron[0])**2+(media[1]-patron[1])**2)]) #ver para mas dimensiones
                distanciasPatronMedias.append(np.sqrt(np.sum((media-patron)**2)))
                #print(f"dist: {distanciasPatronMedias}")
            min_idx = np.argmin(distanciasPatronMedias)
            #print(f"min dist: {min_idx}")
            asignaciones.append(min_idx) #este vector me dice para cada patron, que media le corresponde
        
        for idx_media in range(medias.shape[0]):
            
            #patronesMediaI = [p for i,p in enumerate(asignaciones) if i==idx_media] #ver q pasa si no encuentra nada
            patronesMediaI = []

            for i in range(len(asignaciones)):
                if(asignaciones[i] == idx_media):
                    patronesMediaI.append(i)

            m_inputs_media = m_inputs[patronesMediaI] 
            #print(f"medias: {m_inputs_media} | patrones: {patronesMediaI}")

            #Si no hubo nuevas asignaciones en este conjunto, no cambiarlo
            if(len(patronesMediaI)==0):
                x1_prom = medias[idx_media,0]
                x2_prom = medias[idx_media,1]
            else: #Si hubo cambios en el conjunto, recalcular media
                x1_prom = np.mean(m_inputs_media[:,0])
                x2_prom = np.mean(m_inputs_media[:,1])                
            
            medias[idx_media,:] = [x1_prom, x2_prom]

        #print(asignaciones)

    #Tenemos todas las medias, hay que calcular las salidas de las gaussianas. 
    m_inputs_perceptron = np.zeros((m_inputs.shape[0],neuronasRadiales))
    for idx_p,p in enumerate(m_inputs):
        for idx_m,m in enumerate(medias):
            m_inputs_perceptron[idx_p,idx_m] = utils.gaussiana(p,m,sigma)

    epocas_convergencia_iteracion = nnMultiCapa.Train(m_inputs_perceptron,v_labels, max_epochs=1000, tol_error=.001)
    #print("EPOCAS convg: ", epocas_convergencia_iteracion)

    #TERMINA ENTRENAMIENTO
    #------------------------------------------------------------------------------------------------------------------------
    #TESTEO

    #Feed fodward capa radial
    m_inputs_test = np.ones((datosTest.shape[0],neuronasRadiales))
    for idx_p_test,p_test in enumerate(datosTest[:,:-1]):
        for idx_m,m in enumerate(medias):
            m_inputs_test[idx_p_test,idx_m] = utils.gaussiana(p_test,m,sigma)

    #Feed fodward perceptrones simples
    resultados = np.zeros(datosTest[:,-1:].shape)
    for idx_p, p_test in enumerate(m_inputs_test):
        resultados[idx_p]=nnMultiCapa.Test(p_test)[0,0]

    errores = (resultados - datosTest[:,-1:])**2 
    print("ERROR: " , np.mean(errores)/2)

    v_true = []
    v_false = []
    for i in range(len(m_inputs_test)):
        if(resultados[i] == 1):
            v_true.append(i)
        else:
            v_false.append(i)


    plt.scatter(datosTest[v_true,0], datosTest[v_true,1], color=(1,0,0), label="Verdadero")
    plt.scatter(datosTest[v_false,0], datosTest[v_false,1], color=(0,0,1), label="Falso")

    plot_circles = True
    if(plot_circles):
        for m in medias:
            circle = plt.Circle((m[0], m[1]), sigma, fill=False, edgecolor=(0,0,0), linewidth='1')
            plt.gca().add_patch(circle)
    else:
        plt.scatter(medias[:,0], medias[:,1], color=(0,0,0), label="Centroides")


    plt.title("XOR RBF")
    plt.show() 



if __name__ == "__main__":
    main()