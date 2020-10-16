import numpy as np
import matplotlib.pyplot as plt

def main():

    #Mascara para los ejercicios
    ejercicios = [1, 0, 0]

    #----------------- Ejercicio 1 -----------------
    # Ejemplo de conjunto trapezoidal y gaussiano con sus grados de pertenencias
    if(ejercicios[0]):
        print("Trapecio")
        conj_trap = [-4, -3, -1, 2]
        xs_trap = [-5, -3.5, -2, 1, 4]

        for x in xs_trap:
            print(f"X: {x:+4.2f} | u(x): {grado_membresia(conj_trap, x):+4.2f}")


        print("Gaussiana")
        conj_gauss = [1, 1]
        xs_gauss = [-5, -1, 0, 1, 2, 4.5]

        for x in xs_gauss:
            print(f"X: {x:+4.2f} | u(x): {grado_membresia(conj_gauss, x):+4.2f}")


    #----------------- Ejercicio 2 -----------------
    # Ejemplos de matrices de conjuntos trapezoidales y gaussianos
    # Graficas de cada uno de los conjuntos
    #Trapecios
    M_trap = np.array([[-20, -20, -10, -5],
                       [-10, -5, -5, -2],
                       [-5, -2, -2, 0],
                       [-2, 0, 0, 2],
                       [0, 2, 2, 5],
                       [2, 5, 5, 10],
                       [5, 10, 20, 20]])
    #Gaussianas
    M_gauss = np.array([[-13, 3],
                        [-5, 2],
                        [-2, 1],
                        [0, 1],
                        [2, 1],
                        [5, 2],
                        [13, 3]])
    if(ejercicios[1]):
        rango_x = [-20, 20]

        graficar_conjuntos(M_trap, rango_x)

        graficar_conjuntos(M_gauss, rango_x)


    #----------------- Ejercicio 3 -----------------
    #Fuzzificacion (trabaja con las matrices definidas en el ejercicio 2)
    if(ejercicios[2]):
        xs = [-15, -8, -5, -2, 0, 3, 8, 11]
        print("Fuzzificacion Trapezoidal")
        for x in xs:
            print(f"Fuzzificacion de {x} -> {fuzzificacion(M_trap, x)}")

        print("Fuzzificacion Gaussiana")
        for x in xs:
            print(f"Fuzzificacion de {x} -> {fuzzificacion(M_gauss, x)}")




# Calcula el grado de membresía de 'x' en el conjunto 'conj'
# 'conj' puede ser Gaussiano (2 elementos, media y varianza) o trapezoidal (4 elementos)
def grado_membresia(conj, x):
    tam_conj = len(conj)
    if(tam_conj < 2 or tam_conj > 4 or tam_conj == 3):
        print(f"Error en el tamaño del conjunto ({tam_conj}) debe ser de 2 (gaussiano) o 4 (trapecio) elementos")
    elif tam_conj == 2: #Conjunto Gaussiano
        media = conj[0]
        sigma = conj[1]
        return np.exp(-0.5 * ((x - media)/sigma)**2)
    else: #Conjunto trapezoidal
        a = conj[0]
        b = conj[1]
        c = conj[2]
        d = conj[3]

        # Se inicializa membresia en cero. Se usa una matriz de 1x1 si `x` es un escalar
        x_shape = np.shape(x)
        if x_shape == ():
            membresia = np.zeros((1, 1))
        else:
            membresia = np.zeros(x_shape) 

        # Mascara que define tramos de la funcion
        m = np.vstack((
            np.logical_and(x >= a, x < b),   # Parte creciente
            np.logical_and(x >= b, x <= c),  # Parte constante
            np.logical_and(x > c, x <= d)    # Parte decreciente 
        ))

        # Resultados de aplicar la funcion de cada tramo a x
        y = np.vstack((
            (x - a) / (b - a),
            np.ones(np.shape(x)),
            1 - (x - c) / (d - c)
        ))

        # Se suma la funcion de cada tramo. Aplica solo uno a la vez
        # gracias a la mascara
        membresia += m[0]*y[0] + m[1]*y[1] + m[2]*y[2]

        # Si x es un escalar, se devuelve el unico elemento de la matriz
        if x_shape == ():
            return membresia[0][0]
        else:
            return membresia

#Grafica los 'p' conjuntos que tiene M (las filas)
# M puede tener 4 columnas por conjunto (trapezoidal) o 2 columnas por conjunto (gaussiano)
# 'rango_x' tiene 2 valores, el mínimo y máximo del dominio
def graficar_conjuntos(M, rango_x):

    #Cantidad de elementos por conjunto
    tam_conj = M.shape[1]

    #Los conjuntos tienen que tener 2 o 4 elementos, sino tiro error
    if(tam_conj < 2 or tam_conj > 4 or tam_conj == 3):
        print(f"Error en el tamaño del conjunto ({tam_conj}) debe ser de 2 (gaussiano) o 4 (trapecio) elementos")

    #Grafico cada conjunto de la matriz de conjuntos 'M'
    for p in M:
        if(tam_conj==4): #Conjuntos trapezoidales
            graficar_trapecio(p)
            plt.title("Conjuntos trapezoidales")
        else: #Conjuntos gaussianos
            graficar_gaussiana(p, rango_x)
            plt.title("Conjuntos gaussianos")

    plt.show()

#Grafica las 3 rectas del trapecio a partir de los 4 elementos que definen al conjunto
# 'c' es el color en RGB
def graficar_trapecio(p):
    #Elementos del trapecio
    a = p[0]
    b = p[1]
    c = p[2]
    d = p[3]

    #Grado de membresía de elementos del trapecio
    ua = grado_membresia(p,a)
    ub = grado_membresia(p,b)
    uc = grado_membresia(p,c)
    ud = grado_membresia(p,d)

    plt.plot([a, b, c, d], [ua, ub, uc, ud])


#Grafica una gaussiana a partir de los 2 elementos del conjunto
def graficar_gaussiana(p, rango_x):

    #Rango de variable 'x'
    xs = np.linspace(rango_x[0], rango_x[1], 200)

    #Grado de membresía para cada 'x' en 'xs'
    us = grado_membresia(p, xs)

    plt.plot(xs, us)

#Calcula el grado de membresía de 'x' en cada conjunto de 'M'
def fuzzificacion(M, x):
    #Vector de membresias para 'x'
    #v_membresia = np.zeros(M.shape[0])
    v_membresia = []

    #Calculo el grado de membresía de 'x' en cada conjunto 'p' de 'M'
    for p in M:
        v_membresia.append(grado_membresia(p, x))

    return v_membresia

if __name__ == "__main__":
    main()