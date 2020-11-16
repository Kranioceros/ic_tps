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
    def __init__(self, alfa0, phi0, psi0, nvent, ancho_ventanas, umbral, m_3d):
        self.alfa = alfa0
        self.phi = phi0
        self.psi = psi0
        self.umbral = umbral
        self.nvent = nvent
        self.ancho_ventanas = ancho_ventanas
        self.m_3d = m_3d

    # TODO: Usar el beta (dificultad del item en cuestion)
    def prox_revision(self, sched_idx, t_idx):
        from utils import PrLogistica

        prlog_args = {
            'a': self.alfa,
            'd': 1,
            'phi': self.phi,
            'psi': self.psi,
            'm_3d': self.m_3d
        }

        #-----  Biseccion  ---------
        #Dominio
        t_aux = np.linspace(0, 3600*24*15, 100)

        #Parametros
        tol = 0.05
        max_iter = 10
        N = t_aux.size
        it = 0

        #maximos y minimos
        min_idx = N-1
        y_min =  PrLogistica(**prlog_args, sched=sched_idx, t_actual=min_idx, nvent=self.nvent, ancho_ventanas=self.ancho_ventanas)
        max_idx = t_idx
        y_max =  PrLogistica(**prlog_args, sched=sched_idx, t_actual=max_idx, nvent=self.nvent, ancho_ventanas=self.ancho_ventanas)
        
        #Indice e imagen actual
        idx = int((min_idx+max_idx) / 2)
        y = PrLogistica(**prlog_args, sched=sched_idx, t_actual=idx, nvent=self.nvent, ancho_ventanas=self.ancho_ventanas)

        #Error actual
        err = self.umbral - y

        while it < max_iter and abs(err) > tol:
            if err < 0:
                max_idx = idx
                idx = int((min_idx + idx) / 2)
            else:
                min_idx = idx
                idx = int((max_idx + idx) / 2)
            it += 1
            y = PrLogistica(**prlog_args, sched=sched_idx, t_actual=idx, nvent=self.nvent, ancho_ventanas=self.ancho_ventanas)
            err = self.umbral - y
        
        return (t_aux[idx], y)