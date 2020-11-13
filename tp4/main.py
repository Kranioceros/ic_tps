import numpy as np
from GA import GA
import matplotlib.pyplot as plt
from functools import reduce

def main():

    N = 30 #Cantidad de agentes en una poblacion
    n = 30 #Cantidad de alelos en cada agente
    probCrossOver = 0.9 #Probabilidad de que haya cruza
    probMutation = .1 #Probabilidad de que un alelo se mute
    maxGens = 1000 #Cantidad maxima de generacion a iterar

    #Pares de decodificadores/fitness para cada funcion
    ejemplos = [(DecoDecimal1, fitness1, -512, 512, derivada1),(DecoDecimal2, fitness2, 0, 20, derivada2), (DecoDecimalDoble, fitness3, -100, 100, derivada3_x)] 

    #Cambiar esta variable para cambiar de ecuacion ejemplo
    # Numero de la ecuacion usada (0,1,2)
    ec = 2

    #Funcion de decodificacion y fitness a usar
    deco = ejemplos[ec][0]
    fitn = ejemplos[ec][1]
    a = ejemplos[ec][2]
    b = ejemplos[ec][3]
    deriv = ejemplos[ec][4]

    #Creo el controlador de la poblacion
    ga = GA(N, n, probCrossOver, probMutation, deco, fitn, maxGens)

    #Evoluciono
    #convGen -> si durante convGen generaciones seguidas se repite el mejor fitness -> termino
    ga.Evolve(elitismo=True, convGen=100)


    #------ Graficas de mejor solucion por generacion ---------
    if(ec<2):
        xs = np.linspace(a, b, 1000)
        plt.plot(xs, -fitn(xs), color='b')
        plt.scatter(ga.bestAgentsX, ga.bestAgentsY, color='r')
        plt.title("Método evolutivo")

        #Gradiente
        #Numero al azar entre a y b
        x0 = np.interp(np.random.random(), [0,1], [a,b])
        xs_deriv = []
        ys_deriv = []
        lr = 0.1
        for _i in range(1,100):
            xs_deriv.append(x0)
            ys_deriv.append(-fitn(x0))
            x0 -= lr*deriv(x0)
        plt.figure(2)
        plt.plot(xs, -fitn(xs), color='b')
        plt.scatter(xs_deriv, ys_deriv, color='r')
        plt.title("Método Gradiente")
    else:
        xs = np.linspace(-100, 100, 30)
        ys = np.linspace(-100, 100, 30)

        X, Y = np.meshgrid(xs,ys)

        Z = -fitn((X,Y))
        
        xs_ys = reduce(separar, ga.bestAgentsX)

        bestXs = xs_ys[0]
        bestYs = xs_ys[1]

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(bestXs, bestYs, ga.bestAgentsY, color='r')
        ax.plot_wireframe(X, Y, Z, color=(0.8,0.8,0.8))
        plt.title("Método Evolutivo")

        #Gradiente
        #Numero al azar entre a y b
        x0 = np.interp(np.random.random(), [0,1], [a,b])
        y0 = np.interp(np.random.random(), [0,1], [a,b])
        xs_deriv = []
        ys_deriv = []
        zs_deriv = []
        lr = 0.3
        for _i in range(1,3000):
            xs_deriv.append(x0)
            ys_deriv.append(y0)
            zs_deriv.append(-fitn((x0,y0)))
            x0 -= lr*derivada3_x(x0,y0)
            y0 -= lr*derivada3_y(x0,y0)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(xs_deriv, ys_deriv, zs_deriv, color='r')
        ax.plot_wireframe(X, Y, Z, color=(0.8,0.8,0.8))
        plt.title("Método Gradiente")

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
    if type(x[0]) is np.float64 and type(y[0]) is np.float64:
        (x1,y1) = x
        (x2,y2) = y
        return ([x1,x2], [y1,y2])
    elif type(x[0]) is list and type(y[0]) is np.float64:
        (xs, ys) = x
        (nx, ny) = y
        xs.append(nx)
        ys.append(ny)
        return ((xs,ys))

def derivada1(x):
    aux = np.sqrt(np.abs(x))
    return -np.sin(aux) - (x**2*np.cos(aux))/(2*np.abs(x)*aux)

def derivada2(x):
    return 1 + 15*np.cos(3*x) - 40*np.sin(5*x)

def derivada3_x(x,y):
    aa = 10*x*np.sin(100*(x**2+y**2)**0.1) / ((x**2+y**2)**0.65)
    bb = 0.5*x*np.sin(50*(x**2+y**2)**0.1)**2 / ((x**2+y**2)**0.75)
    return aa + bb

def derivada3_y(x,y):
    aa = 10*y*np.sin(100*(x**2+y**2)**0.1) / ((x**2+y**2)**0.65)
    bb = 0.5*y*np.sin(50*(x**2+y**2)**0.1)**2 / ((x**2+y**2)**0.75)
    return aa + bb

if __name__ == "__main__":
    main()