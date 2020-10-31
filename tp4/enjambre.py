import numpy as np
from particula import particula

def main():

    N = 100  #tamaño del enjambre
    maxIter = 100 #cantidad maxima de iteraciones

    #Para probar distintas ecuaciones:
        #Cambiar el primer argumento de particula por la funcion de fitness que se quiera usar
        #Cambiar dim por lo que sea adecuado
        #Cambiar los xmin y xmax

    dim = 1 #dimension del problema
    enjambre = []
    for _i in range(N):
        enjambre.append(particula(fitness1, dim, xmin=-512, xmax=512))

    bestGlobalY = 99999
    bestGlobalX = np.zeros(dim)

    bestGlobalPrevio = bestGlobalY
    bestRepetido = 0
    maxRepetido = 10

    for _i in range(maxIter):
        for part in enjambre:
            fit = part.Evaluar()
            part.ActualizarMejor(fit)

            if(fit < bestGlobalY):
                bestGlobalY = fit
                bestGlobalX = part.x
                bestRepetido = 0
                bestGlobalPrevio = bestGlobalY

        if(bestGlobalPrevio == bestGlobalY):
            bestRepetido += 1
            if(bestRepetido > maxRepetido):
                print(f"Convergencia por repeticion en iteracion {_i}")
                break

        for part in enjambre:
            part.ActualizarPosicion()
            part.ActualizarVelocidad(bestGlobalX)

    print(f"Best Global X: {bestGlobalX} | Best Global Y: {bestGlobalY}")


def fitness1(x):
    return -x*np.sin(np.sqrt(np.abs(x)))

def fitness2(x):
    return x + 5*np.sin(3*x) + 8*np.cos(5*x)

def fitness3(xy):
    x = xy[0]
    y = xy[1]

    return (x**2 + y**2)**0.25 * (np.sin(50*(x**2+y**2)**0.1)**2 + 1)

if __name__ == "__main__":
    main()