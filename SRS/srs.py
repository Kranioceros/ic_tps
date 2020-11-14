import numpy as np

class SRS:
    def __init__(self):
        print('Algo esta muy mal con tu SRS')
        pass

    # Devuelve una tupla (estado, t_revision)
    def prox_revision(self, ts, cs, ss):
        print('Algo esta muy mal con tu SRS')
        return None

class Uniforme(SRS):
    def __init__(self, t):
        self.t = t

    def prox_revision(self, ts, cs, ss):
        return ts[-1] + self.t