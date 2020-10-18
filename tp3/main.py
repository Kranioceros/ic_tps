import numpy as np
import matplotlib.pyplot as plt

def main():

    #Mascara para los ejercicios
    ejercicios = [1, 1, 1, 1, 1, 1]

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


    #----------------- Ejercicio 4 -----------------
    S = np.array([[-7, -5, -5, -3],
                      [-5, -3, -3, -1],
                      [-3, -1, -1, 0],
                      [-1, 0, 0, 1],
                      [0, 1, 1, 3],
                      [1, 3, 3, 5],
                      [3, 5, 5, 7]], dtype='f')

    if(ejercicios[3]):
        a = np.array([0, 0.7, 0.3, 0, 0, 0, 0])

        print(f"Defuzz: {defuzzificacion(S, a)}")
        graficar_conjuntos(S, [-20,20], a)


    #----------------- Ejercicio 5 -----------------
    if(ejercicios[4]):
        r = [0,1,2,3,4,5,6]

        print(f"Defuzz_regla: {defuzzificacion_regla(M_trap,S,r,5)}")

    #----------------- Ejercicio 6 -----------------
    if(ejercicios[5]):
        r = [0,1,2,3,4,5,6]
        #r = [6,5,4,3,2,1,0]
        #r = [2,3,6,0,4,1,5]
        
        xs = np.linspace(-20, 20, 200)

        ys = []
        for x in xs:
            ys.append(defuzzificacion_regla(M_gauss,S,r,x))

        plt.plot(xs,ys)
        plt.show()

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
        # Volvemos 0 los valores sin sentido
        y[np.logical_or(np.isnan(y), np.isinf(y))] = 0

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
def graficar_conjuntos(M, rango_x, pesos=()):

    if(len(pesos)==0):
        pesos = np.ones(M.shape[0])

    #Cantidad de elementos por conjunto
    tam_conj = M.shape[1]

    #Los conjuntos tienen que tener 2 o 4 elementos, sino tiro error
    if(tam_conj < 2 or tam_conj > 4 or tam_conj == 3):
        print(f"Error en el tamaño del conjunto ({tam_conj}) debe ser de 2 (gaussiano) o 4 (trapecio) elementos")

    #Grafico cada conjunto de la matriz de conjuntos 'M'
    for (idx,p) in enumerate(M):
        if(tam_conj==4): #Conjuntos trapezoidales
            graficar_trapecio(p, pesos[idx])
            plt.title("Conjuntos trapezoidales")
        else: #Conjuntos gaussianos
            graficar_gaussiana(p, rango_x)
            plt.title("Conjuntos gaussianos")

    plt.show()

#Grafica las 3 rectas del trapecio a partir de los 4 elementos que definen al conjunto
# 'c' es el color en RGB
def graficar_trapecio(p, peso=1):
    #Elementos del trapecio
    a = p[0]
    b = p[1]
    c = p[2]
    d = p[3]

    #Grado de membresía de elementos del trapecio
    ua = grado_membresia(p,a)*peso
    ub = grado_membresia(p,b)*peso
    uc = grado_membresia(p,c)*peso
    ud = grado_membresia(p,d)*peso

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
    # Funcion auxiliar que devuelve la a `conj` de todos los valores
    # de x
    def membresia_x(conj):
        return grado_membresia(conj, x)

    return np.apply_along_axis(membresia_x, 1, M)

# Calcula el area y centroide de un conjunto borroso
def area_centroide(conj, peso=1):
    tam_conj = len(conj)
    if(tam_conj < 2 or tam_conj > 4 or tam_conj == 3):
        print(f"Error en el tamaño del conjunto ({tam_conj}) debe ser de 2 (gaussiano) o 4 (trapecio) elementos")
    elif tam_conj == 2: #Conjunto Gaussiano
        cent = conj[0]
        area = conj[1]*np.sqrt((2*np.pi))
        return (area, cent)
    else: #Conjunto trapezoidal
        a = conj[0]
        b = conj[1]
        c = conj[2]
        d = conj[3]
        # Centros de gravedad de las tres partes del trapecio
        cg1 = (b+c)/2
        cg2 = b - (b-a) / 3
        cg3 = c + (d-c) / 3
        # Areas de las tres partes del trapecio
        ar1 = (b-a)*grado_membresia(conj, b)*peso / 2    # Primer triangulo
        ar2 = (c-b)*(grado_membresia(conj, c))*peso      # Rectangulo
        ar3 = (d-c)*(grado_membresia(conj, c))*peso / 2  # Segundo traingulo

        area = ar1 + ar2 + ar3

        if(area == 0):
            cent = 0
        else:
            cent = (ar1*cg1 + ar2*cg2 + ar3*cg3) / area
        return (area, cent)
    
    pass

def defuzzificacion(S, a, r=()):
    if(len(r)==0):
        r = np.arange(len(a))

    v_areas_cent = []
    for idx in range(S.shape[0]):
        v_areas_cent.append(area_centroide(S[r[idx],:], a[idx]))

    num = 0
    den = 0
    for i in v_areas_cent:
        num += i[0]*i[1]
        den += i[0]

    return num/den


def defuzzificacion_regla(M, S, r, x):
    
    a = fuzzificacion(M, x)
    
    return defuzzificacion(S, a, r)


if __name__ == "__main__":
    main()