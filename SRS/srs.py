import numpy as np
from utils import integral_acumulada
from tqdm import tqdm

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
    m_acum_cs = None
    m_acum_ss = None

    @classmethod
    # Recibe la resolucion `res` con la cual se calculan las funciones acumuladas
    # Recibe el sigma (en minutos) con el cual se calcula la densidad
    # Recibe cs y ss de todos los schedules
    def init_acums(cls, m_t, m_c, m_s, res=1000, sigma=30):
        print("---INICIA init_acums---")
        # Nos fijamos si no calculamos previamente estos acumulados
        c_fname = "acum_c-res" + str(res) + "-sigma" + str(sigma)
        s_fname = "acum_s-res" + str(res) + "-sigma" + str(sigma)
        # Cargamos el archivo si existe
        # TODO
        # Inicializamos acum_cs y acum_ss
        S = m_c.shape[0]
        v_t = np.linspace(0, 15*24*3600, res)
        m_acum_cs = np.zeros(S, res)
        m_acum_ss = np.zeros(S, res)
        for s in range(S):
            m_acum_cs = integral_acumulada(v_t, m_t[s], m_c[s], sigma)
            m_acum_ss = integral_acumulada(v_t, m_t[s], m_s[s], sigma)
            # CONTINUARA


    def __init__(self, alfa0, phi0, psi0, umbral):
        self.alfa = alfa0
        self.phi = phi0
        self.psi = psi0
        self.umbral = umbral

    # TODO: Usar el beta (dificultad del item en cuestion)
    def prox_revision(self, delta, t_ult_rev, acum_cs, acum_ss):
        from utils import PrLogisticaOpt

        if acum_cs.size != acum_ss.size:
            print('Flasheaste cualquiera hermano')
            return

        #PrLogisticaOpt(a, d, phi, psi, t_ult_rev, t, estudioAcum_cs, estudioAcum_ss)

        idx_ult_rev = int(np.ceil(t_ult_rev * acum_cs.size / (15*24*3600)))

        prlog_args = {
            'a': self.alfa,
            'd': delta,
            'phi': self.phi,
            'psi': self.psi,
            't_actual': t_ult_rev,
            'estudioAcum_cs': acum_cs,
            'estudioAcum_ss': acum_ss,
        }

        #-----  Biseccion  ---------
        #Parametros
        tol = 0.05
        max_iter = 10
        N = acum_ss.size
        it = 0

        # y_min, y_max: Valores de probabilidad minimos y maximos hasta ahora
        # min_idx, max_idx: Indices correspondientes a minimo y maximo
        min_idx = N-1
        max_idx = idx_ult_rev
        
        #Indice, tiempo e imagen
        idx = int((min_idx+max_idx) / 2)
        t = idx * (15*24*3600) / N
        y = PrLogisticaOpt(**prlog_args, t=(t_ult_rev + 15*24*3600) / 2)

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
            # Obtenemos el t real a partir del indice
            t = idx * (15*24*3600) / N
            # Evaluamos la probabilidad en el punto
            y = PrLogisticaOpt(**prlog_args, t=t)
            # Calculamos el error
            err = self.umbral - y

        return (t, y)