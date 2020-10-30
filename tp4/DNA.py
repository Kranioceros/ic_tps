import numpy as np
#Clase que define a un agente para usar en un algoritmo genetico
# n-> cantidad de alelos
class DNA:

    def __init__(self, n):
        self.fitness = 0.0
        self.fitnessNormalize = 0.0

        self.n = n

        self.dna = np.zeros(self.n)

        self.Initialize()

    #Inicializar agente al azar
    #TODO: seguro hay una forma mas eficiente de hacer un vector de 1s y 0s al azar
    def Initialize(self):
        #creo un vector con valores al azar entre -0.5 y 0.5
        self.dna = np.random.rand(self.n)-0.5

        #A los positivos los hago 1 
        self.dna[self.dna>=0] = 1

        #A los negativos los hago 0
        self.dna[self.dna<0] = 0   


    #Combina este agente con el agente 'a'
    #prob -> probabilidad de que se genere la cruza
    def CrossOver(self, a, prob):
        #Nuevo agente
        newAgent1 = DNA(self.n)
        newAgent2 = DNA(self.n)

        #"Tiro la moneda"
        crossover = np.random.rand()

        #Si estoy dentro de la probabilidad de cruza
        if(crossover <= prob):
            #Punto de corte
            crossPoint = np.random.randint(0, self.n)
            
            #Completo la primera parte con lo del padre 1
            newAgent1.dna[0:crossPoint] = self.dna[0:crossPoint]
            #Completo la segunda parte con lo del padre 2
            newAgent1.dna[crossPoint:] = a.dna[crossPoint:]
            
            #Completo la primera parte con lo del padre 2
            newAgent2.dna[0:crossPoint] = a.dna[0:crossPoint]
            #Completo la segunda parte con lo del padre 1
            newAgent2.dna[crossPoint:] = self.dna[crossPoint:]
        else:
            #No hay cruza, los nuevos agentes son iguales a los padres
            newAgent1.dna = self.dna
            newAgent2.dna = a.dna

        #Devuelvo los nuevos agentes
        return (newAgent1, newAgent2)


    #Muta al agente
    #TODO: Se podría dividir la probabilidad de mutacion entre todos los alelos y "tirar una moneda" para cada uno
    def Mutate(self, prob, perAllele=False):
        #Distribuyo la probabilidad de mutacion entre todos los alelos
        if(perAllele):
            #Probabilidad de mutacion de cada alelo
            probPerAllele = prob/self.n

            #Para cada alelo
            for i in range(self.n):
                #"Tiro la moneda"
                mutate = np.random.rand()

                #Si estoy dentro de la probabilidad
                if(mutate <= probPerAllele):
                    #Invierto bit
                    self.dna[i] = not(self.dna[i])
        #Solo un alelo es mutado
        else:
            #Agarro un alelo al azar
            randAllele = np.random.randint(0, self.n)

            # "Tiro la moneda"
            mutate = np.random.rand()

            # Si estamos dentro de la probabilidad de mutacion, muto
            if(mutate <= prob):
                #Invierto bit
                self.dna[randAllele] = not(self.dna[randAllele])