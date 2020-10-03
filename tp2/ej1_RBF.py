import numpy as np
import matplotlib.pyplot as plt
import math

#Primera parte de la RBF

#inicializamos las medias de las gausianas
#para inicializar las medias, o formo k grupos aleatorios, o agarro k patrones y los usio como medias iniciales
#1 por cada patron del archivo, calculamos la distancia a cada media de las neuronas.
#2 asigno cada patron al grupo o neurona que tiene la media mas cercana al mismo
#3 con los patrones asignados a cada media, recalculo las medias existentes.
#4 si no hay mas reasignaciones, corto
#5 vuelvo al paso 1


def main():
    neuronas = 6
    datos = np.genfromtxt("datos/XOR_trn.csv", dtype=float, delimiter=',')
    #Matriz de patrones sin etiquetas
    m_inputs = datos[:,:-1]
    #Vector de etiquetas de los patrones
    v_labels = datos[:, -1]

    idx = np.arange(m_inputs.shape(0))
    dimension = m_inputs.shape(1)

    medias = []
    
    np.random.shuffle(idx)
    for i in range(neuronas):
        medias.append(m_inputs[i])


    asignaciones_ant = []
    asignaciones = []

    while(asignaciones != asignaciones_ant):
        asignaciones_ant = asignaciones
        asignaciones = []
        for patron in m_inputs:
            distanciasPatronMedias = []
            for media in medias:
                distanciasPatronMedias.append([math.sqrt((media[0]-patron[0])**2+(media[1]-patron[1])**2)]) #ver para mas dimensiones
            max_idx = np.argmax(distanciasPatronMedias)
            asignaciones.append(max_idx) #este vector me dice para cada patron, que media le corresponde
        
        for i in range(medias):
            patronesMediaI = [i for i,x in enumerate(asignaciones) if x==i] #ver q pasa si no encuentra nada
            m_inputs_media = m_inputs[patronesMediaI] 
            x1_prom = np.mean(m_inputs_media[:,0])
            x2_prom = np.mean(m_inputs_media[:,1])
            medias[i] = [x1_prom, x2_prom]


if __name__ == "__main__":
    main()
