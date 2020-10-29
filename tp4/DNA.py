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
    #prct indica el porcentaje que agarramos de aeste agente y 1-prct es lo del agente 'a'
    #TODO: ver si a prct lo dejamos en crossover o lo ponemos como un parametro de inicializacion de GA
    def CrossOver(self, a, prct):
        newAgent = DNA(self.n)

        #Cantidad de alelos del padre 1
        Na1 = int(self.n*prct)
        
        #Completo la primera parte con lo del padre 1
        newAgent.dna[0:Na1] = self.dna[0:Na1]
        #Completo la segunda parte con lo del padre 2
        newAgent.dna[Na1:] = a.dna[Na1:]
        
        #Devuelvo el nuevo agente
        return newAgent

    #Muta al agente
    #TODO: Se podría dividir la probabilidad de mutacion entre todos los alelos y "tirar una moneda" para cada uno
    def Mutate(self, prob):
        #Agarro un alelo al azar
        randAllele = np.random.randint(0, self.n)

        # "Tiro la moneda"
        mutate = np.random.rand()

        # Si estamos dentro de la probabilidad de mutacion, muto
        if(mutate < prob):
            #Invierto bit
            self.dna[randAllele] = not(self.dna[randAllele])