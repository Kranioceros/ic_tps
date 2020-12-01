import numpy as np
from tqdm import tqdm
from pathlib import Path

class SRS:
    def __init__(self):
        print('Algo esta muy mal con tu SRS')
        pass

    # Devuelve una tupla (estado, t_revision)
    def prox_revision(self, sched, t_ult_rev):
        print('Algo esta muy mal con tu SRS')
        return None


class Uniforme(SRS):
    def __init__(self, t):
        self.t = t

    def prox_revision(self, _sched, t_ult_rev):
        return (t_ult_rev + self.t,1)


class SM2(SRS):
    def __init__(self, alfa, beta, gamma):
        self.difficulty = None
        self.hist_correct = 0
        self.alfa = alfa
        self.beta = beta
        self.gamma = gamma

    @classmethod
    def init_class(cls,lens, m_t, m_c, m_s, m_d):
        # Copiamos datos
        cls.m_t = m_t
        cls.m_c = m_c
        cls.m_s = m_s
        cls.lens = lens
        cls.m_d = 1.3 + (1-m_d)


    # Devuelve t_revision
    def prox_revision(self, sched, t_ult_rev):
        idx_ult_rev = int(np.ceil(t_ult_rev * SM2.m_c[sched,SM2.lens[sched]-1].size / (15*24*3600)))
        dias = 24 * 3600
        prox_rev = 0 #t_ult_rev
        correct = SM2.m_c[sched,idx_ult_rev]  
        total = SM2.m_s[sched,idx_ult_rev]
        if self.difficulty is None: 
            self.difficulty = SM2.m_d[sched]
        
        # Calculamos performance (va entre 0 y 5)
        performance = (correct / total * 5)

        # Ajustamos dificultad
        self.difficulty += self.alfa + self.beta * performance + self.gamma * performance ** 2

        # Actualizamos numero de correctas consecutivas. Aca consideramos una
        # performance de 3 como "correcto". Esto es para salvar la diferencia
        # con SM2, que considera cada instancia de la palabra como una pregunta,
        # mientras que nosotros trabajamos a nivel de la sesion
        if performance >= 3:
            self.hist_correct += 1
        else:
            self.hist_correct = 0

        # Calculamos tiempo de proxima revision. 
        if performance >= 3:
            prox_rev += (6 * (self.difficulty ** (self.hist_correct - 1)) * dias)
        else:
            prox_rev += 1 * dias

        return (prox_rev,1)


class SRGA(SRS):
    # pylint: disable=unsubscriptable-object, unused-variable
    m_acum_cs = None
    m_acum_ss = None
    m_deltas  = None

    @classmethod
    # Recibe la resolucion `res` con la cual se calculan las funciones acumuladas
    # Recibe el sigma (en minutos) con el cual se calcula la densidad
    # Recibe cs y ss de todos los schedules
    def init_class(cls, lens, m_t, m_c, m_s, m_d, res=1000, sigma=30):
        from utils import integral_acumulada
        print("---INICIA init_acums---")
        # Nos fijamos si no calculamos previamente estos acumulados
        c_fname = "acum_c-res" + str(res) + "-sigma" + str(sigma) + ".npy"
        s_fname = "acum_s-res" + str(res) + "-sigma" + str(sigma) + ".npy"
        # Cargamos el archivo si existe
        c_path = Path('SRS/data/' + c_fname)
        s_path = Path('SRS/data/' + s_fname)
        if c_path.is_file() and s_path.is_file():
            print("Cargando datos cacheados...")
            cls.m_acum_cs = np.load(c_path)
            cls.m_acum_ss = np.load(s_path)
        else:
            print("Generando matrices...")
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
            np.save(c_path, cls.m_acum_cs)
            np.save(s_path, cls.m_acum_ss)

        #nos traemos las dificultades del archivo generado
        cls.m_deltas = m_d


    def __init__(self, alfa0, phi0, psi0, umbral):
        self.alfa = alfa0
        self.phi = phi0
        self.psi = psi0
        self.umbral = umbral

    # TODO: Usar el beta (dificultad del item en cuestion)
    def prox_revision(self, sched, t_ult_rev):
        from utils import PrLogisticaOpt

        acum_cs = SRGA.m_acum_cs[sched]
        acum_ss = SRGA.m_acum_ss[sched]
        delta   = SRGA.m_deltas[sched]

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
        max_iter = 8
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