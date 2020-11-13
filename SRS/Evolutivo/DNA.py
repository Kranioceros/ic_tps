import numpy as np
from debug import dbg
#Clase que define a un agente para usar en un algoritmo genetico
# n-> cantidad de alelos
# dna -> array de 1s y 0s
# fitness -> fitness actual del agente
# fitnessNormaliza -> fitness actual normalizado

#n deberia ser:
    # eg: precision = (10, 5, 4) -> esto serían 3 variables, de 10,5 y 4 bits

class DNA:

    def __init__(self, v_var=(20)):
        self.fitness = 0.0
        self.fitnessNormalize = 0.0

        self.n_var = len(v_var)
        self.v_var = v_var

        #TODO: tirar error si n_var != len(v_precision)

        #La cantidad de bits totales es la suma de las precisiones
        self.n = sum(v_var)
       
        self.dna = np.zeros(self.n)

        self.Initialize()

    #Inicializar agente al azar
    #TODO: forma mas eficiente de inicializar un vector de 0s y 1s??
    def Initialize(self):
        #creo un vector con valores al azar entre -0.5 y 0.5
        self.dna = np.random.rand(self.n)-0.5

        #A los positivos los hago 1 
        self.dna[self.dna>=0] = 1

        #A los negativos los hago 0
        self.dna[self.dna<0] = 0   


    #Combina este agente con el agente 'a'
    #prob -> probabilidad de que se genere la cruza
    #TODO: probar hacer que cada variable tenga su probabilidad de cruza
    #TODO: hacer que se pudean elegir cuántos hijos sacar de la cruza
    def CrossOver(self, a, prob):
        
        v_newChilds = []

        #Se crean dos hijos con la cruza
        newChild1 = DNA(self.v_var)
        newChild2 = DNA(self.v_var)

        v_newChilds.append(newChild1)
        v_newChilds.append(newChild2)


        #"Tiro la moneda"
        do_crossover = np.random.rand()

        #Si estoy por fuera de la probabilidad, devuelvo los hijos iguales a los padres
        if(do_crossover > prob):
            newChild1.dna = np.array(self.dna)
            newChild2.dna = np.array(a.dna)
            return v_newChilds

        #Limites para el punto de corte
        start = 0
        end = self.v_var[0]

        #Por cada variable agarro desde v_precision[p-1] a v_precision[p]
        for p in range(self.n_var):
            #Punto de corte
            crossPoint = np.random.randint(start, end)

            #El hijo1 toma (padre1, padre2)
            newChild1.dna[start:crossPoint] = self.dna[start:crossPoint]
            newChild1.dna[crossPoint:end] = a.dna[crossPoint:end]

            #El hijo2 toma (padre2, padre1)
            newChild2.dna[start:crossPoint] = a.dna[start:crossPoint]
            newChild2.dna[crossPoint:end] = self.dna[crossPoint:end]

            #Nuevos limites para el punto de corte
            if(p >= self.n_var-1):
                break
            start += self.v_var[p]
            end += self.v_var[p+1]

        return v_newChilds


    #Muta al agente
    #TODO: me parece que lo correcto es que solo un bit se mute a lo largo de todo el dna, y no una mutacion por cada variable. sin embargo ayuda a convergencia esto ultimo
    def Mutate(self, prob, allmutate=True):
        
        # "Tiro la moneda"
        mutate = np.random.rand()
        
        #Si estoy por fuera de la probabilidad, no muto
        if(mutate > prob):
            return

        if(allmutate):
            #Limites para el punto de corte
            start = 0
            end = self.v_var[0]

            #Para cada variable
            for p in range(self.n_var):
                #Agarro un alelo al azar
                randAllele = np.random.randint(start, end)

                self.dna[randAllele] = not(self.dna[randAllele])
                
                #Nuevos limites para el punto de corte
                if(p >= self.n_var-1):
                    break
                start += self.v_var[p]
                end += self.v_var[p+1]
        else:
            randAllele = np.random.randint(0,self.dna.size)

            self.dna[randAllele] = not(self.dna[randAllele])