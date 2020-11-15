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

class SRGA(SRS):
    #def PrLogistica(a,d,phi,psi,c,n,ts,t,nvent=5):
    def __init__(self, alfa0, phi0, psi0, nvent, umbral):
        self.alfa = alfa0
        self.phi = phi0
        self.psi = psi0
        self.umbral = umbral
        self.nvent = nvent

    # TODO: Usar el beta (dificultad del item en cuestion)
    def prox_revision(self, ts, cs, ss):
        from utils import biseccion, PrLogistica
        prlog_args = {
            'a': self.alfa,
            'd': 1,
            'phi': self.phi,
            'psi': self.psi,
            'c': cs,
            'n': ss,
            'ts': ts,
        }

        t_aux = np.linspace(0, 3600*24*15, 100)
        (t_p, _p) = biseccion(0.80, 0.1, t_aux,
            lambda t: PrLogistica(**prlog_args, t=t, nvent=self.nvent), max_iter=10)

        return t_p