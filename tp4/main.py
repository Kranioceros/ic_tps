import numpy as np
from GA import GA

def main():

    N = 100 #Cantidad de agentes en una poblacion
    n = 20 #Cantidad de alelos en cada agente
    probCrossOver = 0.9 #Probabilidad de que haya cruza
    probMutation = .1 #Probabilidad de que un alelo se mute
    maxGens = 5000 #Cantidad maxima de generacion a iterar

    #Creo el controlador de la poblacion
    ga = GA(N, n, probCrossOver, probMutation, DecoDecimalDoble, fitness3, maxGens)

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
            fitness += 10
    return fitness

def DecoIdentidad(v):
    return (v,v)


def fitness1(x):
    y = -x*np.sin(np.sqrt(np.abs(x)))
    return -y

def fitness2(x):
    y = x + 5*np.sin(3*x) + 8*np.cos(5*x)
    return -y

def fitness3(xy):
    x = xy[0]
    y = xy[1]

    z = (x**2 + y**2)**0.25 * (np.sin(50*(x**2+y**2)**0.1)**2 + 1)
    return z


def DecoDecimal(v, a=0, b=20):
    k = len(v)
    d = sum(2**(k-np.array(range(1,k+1)))*v)

    x = a + (d*((b-a)/((2**k)-1)))

    return x

def DecoDecimalDoble(v, a=-100, b=100):
    k = int(len(v)/2)

    d1 = sum(2**(k-np.array(range(1,k+1)))*v[0:k])
    d2 = sum(2**(k-np.array(range(1,k+1)))*v[k:])

    x = a + (d1*((b-a)/((2**k)-1)))
    y = a + (d2*((b-a)/((2**k)-1)))

    return (x,y)

if __name__ == "__main__":
    main()