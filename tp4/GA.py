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

        self.bestAgentsX = []
        self.bestAgentsY = []
        self.bestAgentsZ = []

        self.Initialize()


    #Inicializo la poblacion de N agentes al azar
    def Initialize(self):
        #Inicializo N agentes
        for _i in range(self.N):
           self.population.append(DNA(self.n))


    #Controla la logica del algoritmo genetico
    #elitismo -> quedarse con el mejor de cada generacion
    #convGen -> si durante esta cantidad seguida de genraciones se repite el mismo bestFitness -> terminar
    def Evolve(self, elitismo=False, convGen = 10):

        #Cuenta cuantas veces se repite el mejor fitness (en generaciones seguidas)
        bestRepeated = 0

        #El mejor fitness de la generacion anterior
        #TODO: inicializarlo de manera mas inteligente
        bestFitnessPrev = -9999

        #Itero tantas veces como generaciones maximas
        for _i in range(self.maxGens):
            #Poblacion nueva, me voy guardando los nuevos agentes
            newPopulation = []

            #Evaluo poblacion actual y me guardo los fitness y el mejor
            (v_fitness, bestFitnessActual) = self.EvaluatePopulation()

            #El mejor agente de esta poblacion
            bestAgent = self.population[np.argmax(v_fitness)]

            
            best = self.f_deco(bestAgent.dna)
            #if(len(best)>1):
                #self.bestAgentsX.append(best[0])
                #self.bestAgentsY.append(best[1])
            #else:

            self.bestAgentsX.append(best)
            #self.bestAgentsX = np.append(self.bestAgentsX, best)

            self.bestAgentsZ.append(-self.f_fitness(self.f_deco(bestAgent.dna)))
            #self.bestAgentsZ = np.append(self.bestAgentsZ, -self.f_fitness(self.f_deco(bestAgent.dna)))


            #Muestro el mejor fitness para esta generacion
            print(f"Coord Minimas: ({self.f_deco(bestAgent.dna)}, {-self.f_fitness(self.f_deco(bestAgent.dna))})")
            #print(f"max: {np.max(v_fitness)} | mean: {np.mean(v_fitness)} | STD: {np.std(v_fitness)}")

            #Verifico si se repite el mejor fitness
            if(bestFitnessActual == bestFitnessPrev):
                bestRepeated += 1
                #Si ya se repitió convGen veces -> termino
                if(bestRepeated == convGen):
                    print(f"Convergencia por repeticion en generacion {_i}")
                    break
            else:
                #Verifico si el mejor actual es mejor que el anterior
                if(bestFitnessActual > bestFitnessPrev):
                    bestFitnessPrev = bestFitnessActual
                    bestRepeated = 0

            #Ordeno los fitness (solo sirve para el picker ventanas)
            #TODO: sacar esto si no usamos picker ventanas
            v_fitness_ord = -np.sort(-v_fitness)

            #Si hago elitismo me quedo con el mejor de la generacion
            if(elitismo):
                newPopulation.append(bestAgent)

            #Solo para picker ventanas
            #TODO:sacar si no usamos picker ventanas
            i = self.N

            #Itero tantas veces como agentes en una poblacion
            for _j in range(int(self.N/2)):

                #Picker recipiente
                #Eligo dos agentes al "azar"
                a1 = self.Picker()
                a2 = self.Picker()

                #Picker ventanas
                #a1 = self.PickerWindow(i, v_fitness_ord)
                #i -= 1
                #a2 = self.PickerWindow(i, v_fitness_ord)

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
            self.population = list(newPopulation)

            #print(f"Generacion {_i}")
            #self.DebugPopulation()


    #Selecciono al "azar" un agente de toda la poblacion
    #Se va vaciando un recipiente de volumen 1 hasta que llegue a 0 (o menor)
    #El agente que haga que el volumen caiga de 0 es el elegido
    #Se usa el fitness normalizado para ir restando
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


    #Seleccion de agente mediante ventanas
    #TODO: terminarlo
    def PickerWindow(self, i, v_fitness, v_idxs):
        ventana = v_fitness[0:i]

        randIdx = np.random.randint(0, len(ventana))

        return self.population[randIdx]



    #Evalúa cada agente de la poblacion con la funcion de fitness
    #Tambien normaliza todos los fitness
    def EvaluatePopulation(self):
        #Vector de fitness
        v_fitness = np.zeros(self.N)

        #Para cada agente de la poblacion
        for i,a in enumerate(self.population):
            #Calculo su fitness, usando la funcion de decodificacion para pasar el fenotipo
            v_fitness[i] = self.f_fitness(self.f_deco(a.dna))
            a.fitness = v_fitness[i]

        #Busco el mejor fitness
        bestFitness = np.max(v_fitness) #'b'
        worstFitness = np.min(v_fitness) #'a'

        #Mapeo [a,b] -> [0,1]
        #Normalizo los fitness de cada agente con respecto al fitness maximo
        for a in self.population:
            a.fitnessNormalize = (a.fitness-worstFitness)/(bestFitness-worstFitness)

        #Devuelvo el fitness solo como debug, no es necesario devolverlo
        return (v_fitness, bestFitness)


    #Funcion debug para imprimir todos los agentes de la poblacion
    def DebugPopulation(self):
        for i,a in enumerate(self.population):
            print(f"Agente {i}: {a.dna}")
