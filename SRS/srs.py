import numpy as np
from tqdm import tqdm

class SRS:
    def __init__(self):
        print('Algo esta muy mal con tu SRS')
        pass

    # Devuelve una tupla (estado, t_revision)
    def prox_revision(self, delta, sched, t_ult_rev):
        print('Algo esta muy mal con tu SRS')
        return None

class Uniforme(SRS):
    def __init__(self, t):
        self.t = t

    def prox_revision(self, _delta, _sched, t_ult_rev):
        return t_ult_rev + self.t

class SRGA(SRS):
    # pylint: disable=unsubscriptable-object, unused-variable
    m_acum_cs = None
    m_acum_ss = None

    @classmethod
    # Recibe la resolucion `res` con la cual se calculan las funciones acumuladas
    # Recibe el sigma (en minutos) con el cual se calcula la densidad
    # Recibe cs y ss de todos los schedules
    def init_class(cls, lens, m_t, m_c, m_s, res=1000, sigma=30):
        from utils import integral_acumulada
        print("---INICIA init_acums---")
        # Nos fijamos si no calculamos previamente estos acumulados
        _c_fname = "acum_c-res" + str(res) + "-sigma" + str(sigma)
        _s_fname = "acum_s-res" + str(res) + "-sigma" + str(sigma)
        # Cargamos el archivo si existe
        # TODO
        # Inicializamos acum_cs y acum_ss
        S = m_c.shape[0]
        v_t = np.linspace(0, 15*24*3600, res)
        cls.m_acum_cs = np.zeros((S, res))
        cls.m_acum_ss = np.zeros((S, res))
        for s in tqdm(range(S)):
            n = lens[s]
            # pylint: disable=unsubscriptable-object, unsupported-assignment-operation
            cls.m_acum_cs[s] = integral_acumulada(v_t, m_t[s,:n], m_c[s,:n], sigma=sigma*60)
            cls.m_acum_ss[s] = integral_acumulada(v_t, m_t[s,:n], m_s[s,:n], sigma=sigma*60)
        # Guardamos el archivo
        # TODO


    def __init__(self, alfa0, phi0, psi0, umbral):
        self.alfa = alfa0
        self.phi = phi0
        self.psi = psi0
        self.umbral = umbral

    # TODO: Usar el beta (dificultad del item en cuestion)
    def prox_revision(self, delta, sched, t_ult_rev):
        from utils import PrLogisticaOpt

        acum_cs = SRGA.m_acum_cs[sched]
        acum_ss = SRGA.m_acum_ss[sched]

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