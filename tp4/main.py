import numpy as np
from GA import GA
import matplotlib.pyplot as plt
from functools import reduce

#TODO:

def main():

    N = 100 #Cantidad de agentes en una poblacion
    n = 30 #Cantidad de alelos en cada agente
    probCrossOver = 0.9 #Probabilidad de que haya cruza
    probMutation = .1 #Probabilidad de que un alelo se mute
    maxGens = 1000 #Cantidad maxima de generacion a iterar

    #Pares de decodificadores/fitness para cada funcion
    datos = [(DecoDecimal1, fitness1, -512, 512),(DecoDecimal2, fitness2, 0, 20), (DecoDecimalDoble, fitness3, -100, 100)] 

    #Cambiar esta variable para cambiar de ecuacion
    #Para ec=0 y ec=1 cambiar los extremos [a,b] de la funcion DecoDecimal
    #TODO: pensar una forma mas comoda de cambiar entre ecuaciones
    # Numero de la ecuacion usada (0,1,2)
    ec = 2

    #Funcion de decodificacion y fitness a usar
    deco = datos[ec][0]
    fitn = datos[ec][1]
    a = datos[ec][2]
    b = datos[ec][3]

    #Creo el controlador de la poblacion
    ga = GA(N, n, probCrossOver, probMutation, deco, fitn, maxGens)

    #Evoluciono
    #convGen -> si durante convGen generaciones seguidas se repite el mejor fitness -> termino
    ga.Evolve(elitismo=True, convGen=100)


    #------ Graficas de mejor solucion por generacion ---------
    #TODO: El problema esta en clase GA. Lo ideal sería guardarse el X y el Y, sin importar en qué dimensiones estan.
            #Pero no sabia cómo extraer despues la informacion en las tuplas,
                #Por eso ahora en 'bestAgentsX' estan los mejores X
                    #En 'bestAgentsZ' estan las mejoras salidas Y
                        #Y en 'bestAgentsY' estan las coordenadas Y por si es en 3D
    if(ec<2):
        xs = np.linspace(a, b, 1000)
        plt.plot(xs, -fitn(xs), color='b')
        plt.scatter(ga.bestAgentsX, ga.bestAgentsZ, color='r')
    else:
        xs = np.linspace(-10, 10, 1000)
        ys = np.linspace(-10, 10, 1000)

        X, Y = np.meshgrid(xs,ys)

        Z = -fitn((X,Y))
        
        print(f"mejores: {ga.bestAgentsX}")
        xs_ys = reduce(separar, ga.bestAgentsX)
        bestXs = xs_ys[0]
        bestYs = xs_ys[1]

        print(f"xs_ys: {xs_ys}")
        print(f"xs_ys: {bestXs}")
        print(f"xs_ys: {bestYs}")

        #bestXs = ga.bestAgentsX[:,0]
        #bestYs = ga.bestAgentsX[:,1]

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(bestXs, bestYs, ga.bestAgentsZ, color='r')
        ax.plot_wireframe(X, Y, Z, color='b')

    plt.show()


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
def DecoDecimal1(v, a=-512, b=512):
    k = len(v)
    d = sum(2**(k-np.array(range(1,k+1)))*v)

    x = a + (d*((b-a)/((2**k)-1)))

    return x

def DecoDecimal2(v, a=0, b=20):
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

#----- Auxiliares -------------
def separar(x,y):
    if type(x[0]) is int and type(y[0]) is int:
        (x1,y1) = x
        (x2,y2) = y
        return ([x1,x2], [y1,y2])
    elif type(x[0]) is list and type(y[0]) is int:
        (xs, ys) = x
        (nx, ny) = y
        xs.append(nx)
        ys.append(ny)
        return ((xs,ys))


if __name__ == "__main__":
    main()