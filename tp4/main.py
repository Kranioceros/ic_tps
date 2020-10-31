import numpy as np
from GA import GA

def main():

    N = 100 #Cantidad de agentes en una poblacion
    n = 30 #Cantidad de alelos en cada agente
    probCrossOver = 0.9 #Probabilidad de que haya cruza
    probMutation = .1 #Probabilidad de que un alelo se mute
    maxGens = 5000 #Cantidad maxima de generacion a iterar

    #Pares de decodificadores/fitness para cada funcion
    ecuacion_deco = [(DecoDecimal, fitness1),(DecoDecimal, fitness2), (DecoDecimalDoble, fitness3)] 

    #Cambiar esta variable para cambiar de ecuacion
    #Para ec=0 y ec=1 cambiar los extremos [a,b] de la funcion DecoDecimal
    #TODO: pensar una forma mas comoda de cambiar entre ecuaciones
    # Numero de la ecuacion usada (0,1,2)
    ec = 0

    #Funcion de decodificacion y fitness a usar
    deco = ecuacion_deco[ec][0]
    fitn = ecuacion_deco[ec][1]

    #Creo el controlador de la poblacion
    ga = GA(N, n, probCrossOver, probMutation, deco, fitn, maxGens)

    #Evoluciono
    #convGen -> si durante convGen generaciones seguidas se repite el mejor fitness -> termino
    ga.Evolve(elitismo=True, convGen=10)





# --------------- Funciones Fitness --------------------

#Funcion de fitness de prueba, crece segun la cantidad de 1s
def NumberOfOnes(v):
    fitness = 0
    for i in v:
        if i==1:
            fitness += 10
    return fitness

# Ecuacion 1 - N=100, n=30, elitismo=True, convGen=10
def fitness1(x):
    y = -x*np.sin(np.sqrt(np.abs(x)))
    return -y

# Ecuacion 2 - N=100, n=30, elitismo=True, convGen=10
def fitness2(x):
    y = x + 5*np.sin(3*x) + 8*np.cos(5*x)
    return -y

# Ecuacion 3 - N=300, n=50, elitismo=True, convGen=30
def fitness3(xy):
    x = xy[0]
    y = xy[1]

    z = (x**2 + y**2)**0.25 * (np.sin(50*(x**2+y**2)**0.1)**2 + 1)
    return -z


#--------------- Decodificadores -------------------

#Devuelve lo que recibe, porque el fitness de este deco necesita el vector binario
def DecoIdentidad(v):
    return v

#Decodificador binario-decimal de una variable
def DecoDecimal(v, a=0, b=20):
    k = len(v)
    d = sum(2**(k-np.array(range(1,k+1)))*v)

    x = a + (d*((b-a)/((2**k)-1)))

    return x

#Decodificador binario-decimal de dos variables
def DecoDecimalDoble(v, a=-100, b=100):
    k = int(len(v)/2)

    d1 = sum(2**(k-np.array(range(1,k+1)))*v[0:k])
    d2 = sum(2**(k-np.array(range(1,k+1)))*v[k:])

    x = a + (d1*((b-a)/((2**k)-1)))
    y = a + (d2*((b-a)/((2**k)-1)))

    return (x,y)



if __name__ == "__main__":
    main()