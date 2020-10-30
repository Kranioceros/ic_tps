import numpy as np
from GA import GA

def main():

    N = 10 #Cantidad de agentes en una poblacion
    n = 10 #Cantidad de alelos en cada agente
    probCrossOver = .9 #Probabilidad de que haya cruza
    probMutation = .1 #Probabilidad de que un alelo se mute
    maxGens = 500 #Cantidad maxima de generacion a iterar

    #Creo el controlador de la poblacion
    ga = GA(N, n, probCrossOver, probMutation, DecoDecimal, fitness1, maxGens)

    #print("Primera poblacion")
    #ga.DebugPopulation()

    #Evoluciono
    ga.Evolve()

    #print("Ultima poblacion")
    #ga.DebugPopulation()


#Funcion de fitness de prueba, crece segun la cantidad de 1s
def NumberOfOnes(v):
    fitness = 0
    for i in v:
        if i==1:
            fitness += 100
    return fitness

def DecoIdentidad(v):
    return v

def ec1(x):
    return -x*np.sin(np.sqrt(np.abs(x)))

def fitness1(x):
    return 1/x

def ec2(x):
    return x + 5*np.sin(3*x) + 8*np.cos(5*x)

def DecoDecimal(v, a=-512, b=512):
    k = len(v)
    d = sum(2**(k-np.array(range(1,k+1)))*v)

    x = a + (d*((b-a)/(2**k-1)))

    return ec1(x)

if __name__ == "__main__":
    main()