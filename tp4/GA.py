import numpy as np
from DNA import DNA
#Clase que controla el funcionamiento de algoritmo genetico
#Cosas necesarias:
    # f_fitness -> funcion de fitness
    # N -> tamaño de poblacion
    # n -> tamaño de cada agente
    # maxGens -> maxima cantidad de generaciones a iterar
# tener en cuenta que cada vez que se habla de agente, se refiere a una clase DNA, que tiene un vector 'dna' de 1s y 0s
class GA:
    def __init__(self, N, n, probCrossOver, probMutation, f_deco, f_fitness, maxGens):
        
        self.N = N
        self.n = n
        self.probCrossOver = probCrossOver
        self.probMutation = probMutation
        self.f_deco = f_deco
        self.f_fitness = f_fitness
        self.maxGens = maxGens

        self.population = []

        self.Initialize()


    #Inicializo la poblacion de N agentes al azar
    def Initialize(self):
        #Para N agentes
        for _i in range(self.N):
           self.population.append(DNA(self.n))


    def Evolve(self):
        #Itero tantas veces como generaciones maximas
        for _i in range(self.maxGens):
            #Poblacion nueva, me voy guardando los nuevos agentes
            newPopulation = []

            #Evalua poblacion y me guardo los fitness
            v_fitness = self.EvaluatePopulation()

            #Muestro el fitness medio para esta generacion
            print(f"Mean Fitness generacion {_i}: {np.max(v_fitness)}")

            #Itero tantas veces como agentes en una poblacion
            for _j in range(int(self.N/2)):
                #Eligo dos agentes al "azar"
                a1 = self.Picker()
                a2 = self.Picker()

                #Combino los dos agentes y obtengo dos nuevos
                (newAgent1, newAgent2) = a1.CrossOver(a2, self.probCrossOver)

                #Muto a los agentes nuevos
                newAgent1.Mutate(self.probMutation)
                newAgent2.Mutate(self.probMutation)

                #Los agrego a la nueva poblacion
                newPopulation.append(newAgent1)
                #Si N es impar, en la ultima iteracion del for tengo que agregar un solo padre
                if(len(newPopulation) == self.N):
                    break
                newPopulation.append(newAgent2)

            #Una vez generados N agentes nuevos, reemplazo la poblacion actual
            #TODO: revisar si esta asignacion no causa problemas
            self.population = newPopulation


    #Selecciono al "azar" un agente de toda la poblacion
    #Se va vaciando un recipiente de volumen 1 hasta que llegue a 0 (o menor)
    #El agente que haga que el volumen caiga de 0 es el elegido
    #Se usa el fitness normalizado para ir restando
    #TODO: seguro hay una forma de seleccionar un elemento al azar de un vector, en vez de agarrar un indice y despues usarlo
    def Picker(self):
        #Comienzo con el recipiente lleno
        volume = 1.0
        #Variable para escapar del while infinito (puede llegar a pasar que todos los agentes tengan un fitness muy cercano a 0 y tarde demasiado en llegar a 0 el volumen)
        beSafe = 0

        #Indice del agente elegido
        randAgent = 0
        #Mientras que no se vacie el recipiente y estemos dentro del beSafe
        while(volume>=0 and beSafe < 1000):
            #agarro agente al azar
            randAgent = np.random.randint(0,self.N)

            #Resto al volumen actual con respecto al fitness normalizado del agente elegido 
            volume -= self.population[randAgent].fitnessNormalize

            beSafe += 1

        #Devuelvo el agente que vació el recipiente
        return self.population[randAgent]


    #Evalúa cada agente de la poblacion con la funcion de fitness
    #Tambien normaliza todos los fitness
    #TODO: habrá alguna forma de no recorrer dos veces el for??
    def EvaluatePopulation(self):
        #Vector de fitness
        v_fitness = np.zeros(self.N)

        #Para cada agente de la poblacion
        for i,a in enumerate(self.population):
            #Calculo su fitness, usando la funcion de decodificacion para pasar el fenotipo
            v_fitness[i] = self.f_fitness(self.f_deco(a.dna))
            a.fitness = v_fitness[i]

        #Busco el fitness maximo
        maxFitness = np.max(v_fitness)

        #Normalizo los fitness de cada agente con respecto al fitness maximo
        for a in self.population:
            a.fitnessNormalize = a.fitness/maxFitness

        #Devuelvo el fitness solo como debug, no es necesario devolverlo
        return v_fitness


    #Funcion debug para imprimir todos los agentes de la poblacion
    def DebugPopulation(self):
        for i,a in enumerate(self.population):
            print(f"Agente {i}: {a.dna}")