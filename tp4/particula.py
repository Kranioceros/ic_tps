import numpy as np

class particula:
    def __init__(self, f_fitness, dim=1, xmin=0, xmax=1):
        self.x = xmin + np.random.rand(dim) * (xmax-xmin)
        self.v = np.zeros(dim)

        self.dim = dim

        self.xmin = xmin
        self.xmax = xmax

        self.bestY = 99999
        self.bestX = np.array(self.x)

        self.f_fitness = f_fitness


    def Evaluar(self):
        return self.f_fitness(self.x)

    def ActualizarMejor(self, y):
        if(y < self.bestY):
            self.bestY = y
            self.bestX = self.x

    def ActualizarPosicion(self):
        self.x = self.x + self.v

        for i in range(self.dim):    
            if(self.x[i] < self.xmin):
                self.x[i] = self.xmin
            if(self.x[i] > self.xmax):
                self.x[i] = self.xmax


    def ActualizarVelocidad(self, bestGlobalX):
        r1 = np.random.rand(self.dim)*0.1
        r2 = np.random.rand(self.dim)*0.1

        c1 = 1
        c2 = 1

        self.v += c1*r1*(self.bestX - self.x) + c2*r2*(bestGlobalX - self.x)