import numpy as np

#Clase qeu define una particula de un enjambre
#f_fitness -> funcion de "fitness" (en realidad es una funcion de error que hay que minimizar)
#dim -> dimension del problema
#xmin -> cota minima para X
#xmax -> cota maxima para X
#TODO: los xmin y xmax deberían ser vectores que indiquen maximos y minimos para cada variable del problema (para los casos de ejemplos anda como esta)

class particula:
    def __init__(self, f_fitness, dim=1, xmin=0, xmax=1):
        #Inicializo al azar entre xmin y xmax
        self.x = xmin + np.random.rand(dim) * (xmax-xmin)
        #Empiezo sin velocidad
        self.v = np.zeros(dim)

        self.dim = dim

        self.xmin = xmin
        self.xmax = xmax

        self.bestY = 99999
        self.bestX = np.array(self.x)

        self.f_fitness = f_fitness

    #Funcion para evualar la funcion de error
    def Evaluar(self):
        return self.f_fitness(self.x)

    #Actualiza la mejor performance si es necesario
    def ActualizarMejor(self, y):
        if(y < self.bestY):
            self.bestY = y
            self.bestX = self.x

    #Actualiza la posicion actual de la particula
    def ActualizarPosicion(self):
        self.x = self.x + self.v

        #Clipeamos la posicion a las cotas de solucion
        for i in range(self.dim):    
            if(self.x[i] < self.xmin):
                self.x[i] = self.xmin
            if(self.x[i] > self.xmax):
                self.x[i] = self.xmax

    #Actualiza la velocidad actual de la particula
    #Recive la mejor performance del enjambre para usar la componente social
    def ActualizarVelocidad(self, bestGlobalX):
        #Variables de "mutacion" [0, 0.1]
        r1 = np.random.rand(self.dim)*0.1
        r2 = np.random.rand(self.dim)*0.1

        #Constantes de aceleracion
        #TODO: ponerlos como parametros de la particula? o del enjambre? o dejarlos acá?
        c1 = 1
        c2 = 1

        self.v += c1*r1*(self.bestX - self.x) + c2*r2*(bestGlobalX - self.x)