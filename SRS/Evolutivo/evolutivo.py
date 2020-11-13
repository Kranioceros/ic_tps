import numpy as np
from DNA import DNA
from debug import dbg

#Los agentes (DNA) pueden ser cualquier objeto que necesita:
    #En caso de ser genético:
        #un array llamado 'dna' que tiene 'n' elementos
        #'n' se establece en el constructor
        #en el constructor se debe inicializar al azar el 'dna'
        #tener definido una funcion de crossover que reciba otro agente
        #tener una funcion de mutacion

#Clase que controla el funcionamiento de algoritmo genetico
    # N -> tamaño de poblacion
    # n -> tamaño de cada agente
    # probCrossOver -> probabilidad de que suceda una cruza
    # probMutation -> probabilidad de que un agente mute
    # f_deco -> función que pasa de genotipo a fenotipo
    # f_fitness -> funcion de fitness
    # maxGens -> maxima cantidad de generaciones a iterar

class GA:
    def __init__(self, N, v_var, probCrossOver, probMutation, f_deco, f_fitness, maxGens, debugLvl=-1):
        
        self.N = N
        self.probCrossOver = probCrossOver
        self.probMutation = probMutation
        self.f_deco = f_deco
        self.f_fitness = f_fitness
        self.maxGens = maxGens
        self.debugLvl = debugLvl

        self.population = []

        self.Initialize(v_var)


    #Inicializo la poblacion de N agentes al azar
    def Initialize(self, v_var):
        #Inicializo N agentes
        for _i in range(self.N):
           self.population.append(DNA(v_var))


    #Controla la logica del algoritmo genetico
    #elitismo -> quedarse con el mejor de cada generacion
    #convGen -> si durante esta cantidad seguida de genraciones se repite el mismo bestFitness -> terminar
    def Evolve(self, elitismo=True, brecha=.1, convGen = 100):

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

            dbg(f"Mejor agente generacion {_i+1}: {best}", 3, self.debugLvl)

            #Verifico si se repite el mejor fitness
            if(bestFitnessActual == bestFitnessPrev):
                bestRepeated += 1
                #Si ya se repitió convGen veces -> termino
                if(bestRepeated == convGen):
                    dbg(f"Convergencia por repeticion en generacion {_i}", 10, self.debugLvl)
                    break
            else:
                #Verifico si el mejor actual es mejor que el anterior
                if(bestFitnessActual > bestFitnessPrev):
                    bestFitnessPrev = bestFitnessActual
                    bestRepeated = 0

            #Si hago elitismo me quedo con el mejor de la generacion
            if(elitismo):
                newPopulation.append(bestAgent)

            #Brecha generacional
            n_brecha = int(self.N*brecha)
            for i in range(n_brecha):
                newPopulation.append(self.population[i])

            #Itero tantas veces como agentes en una poblacion
            for _j in range(self.N):

                #Picker recipiente
                #Eligo dos agentes al "azar"
                a1 = self.Picker()
                a2 = self.Picker()

                v_newAgents = []
                #Combino los dos agentes y obtengo dos nuevos
                v_newAgents = list(a1.CrossOver(a2, self.probCrossOver))

                dbg(f"Hijos crossover: {len(v_newAgents)}",0,self.debugLvl)

                #Muto a los agentes nuevos
                for ag in v_newAgents:
                    dbg(f"Type: {type(ag)}",0,self.debugLvl)
                    ag.Mutate(self.probMutation)
                    newPopulation.append(ag)

                dbg(f"New population: {len(newPopulation)}", 0, self.debugLvl)

                if(len(newPopulation) >= self.N):
                    break

            dbg(f"Generacion actual: {_i}", 3, self.debugLvl)

            #Una vez generados N agentes nuevos, reemplazo la poblacion actual
            self.population = list(newPopulation[0:self.N])


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

    def DebugPopulation(self):
        for (i,p) in enumerate(self.population):
            dbg(f"Agente {i}: {p.dna} | fitness : {p.fitness} | norm : {p.fitnessNormalize}",1,self.debugLvl)