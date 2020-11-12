import numpy as np
from particula import particula

def main():

    N = 1000  #tamaño del enjambre
    maxIter = 1000 #cantidad maxima de iteraciones

    #Distintos ejemplos para probar
    ejemplos = [(error1, 1, -512, 512), (error2, 1, 0, 20), (error3, 2, -100, 100)]

    #Ejemplo elegido actualmente [0, 1, 2]
    ec = 2

    #Variables por ejemplo (funcion de error, dimension del problema, minimo para x, maximo para x)
    error = ejemplos[ec][0]
    dim = ejemplos[ec][1]
    xmin = ejemplos[ec][2]
    xmax = ejemplos[ec][3]

    #Inicializo el enjambre con particulas posicionadas al azar en el hiperespacio
    enjambre = []
    for _i in range(N):
        enjambre.append(particula(error, dim, xmin, xmax))

    #Mejor global
    bestGlobalY = 99999
    bestGlobalX = np.zeros(dim)

    #Varaibles de corte 
    bestGlobalPrevio = bestGlobalY
    bestRepetido = 0
    maxRepetido = 300


    for _i in range(maxIter):
        #Para cada particula
        for part in enjambre:
            #Evalua su valor de error
            val_error = part.Evaluar()
            #Actaulizo el mejor local de la particula
            part.ActualizarMejor(val_error)

            #Reviso si el performance de esta particula es mejor que el global
            if(val_error < bestGlobalY):
                bestGlobalY = val_error
                bestGlobalX = part.x
                bestRepetido = 0

        #Control de repeticion de mejor global (condicion de corte)
        if(bestGlobalPrevio == bestGlobalY):
            bestRepetido += 1
            if(bestRepetido > maxRepetido):
                print(f"Convergencia por repeticion en iteracion {_i}")
                break

        #Para cada particula
        for part in enjambre:
            #Actualizo posicion y velocidad
            part.ActualizarVelocidad(bestGlobalX)
            part.ActualizarPosicion()
        bestGlobalPrevio = bestGlobalY    

    #Al converger, muestro la mejor global
    if(bestRepetido <= maxRepetido):
        print(f"Convergencia por máxima cantidad de iteraciones {maxIter}")
    print(f"Best Global X: {bestGlobalX} | Best Global Y: {bestGlobalY}")


#------ Funciones de error para cada ejemplo -------------
def error1(x):
    return -x*np.sin(np.sqrt(np.abs(x)))

def error2(x):
    return x + 5*np.sin(3*x) + 8*np.cos(5*x)

def error3(xy):
    x = xy[0]
    y = xy[1]

    return (x**2 + y**2)**0.25 * (np.sin(50*(x**2+y**2)**0.1)**2 + 1)

if __name__ == "__main__":
    main()