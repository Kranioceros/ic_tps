from srs import SRS
import numpy as np

class SM2(SRS):
    # pylint: disable=unsubscriptable-object, unused-variable
    m_t = None
    m_c = None
    m_s = None
    m_d = None

    @classmethod
    def init_class(cls,lens, m_t, m_c, m_s, m_d):
        # Copiamos datos
        cls.m_t = m_t
        cls.m_c = m_c
        cls.m_s = m_s
        cls.lens = lens
        cls.m_d = 1.3 + (1-m_d)

    def __init__(self):
        # TODO: Normalizar delta para que sea siempre >= 1.3. Caso contrario,
        # las constantes utilizadas van a llevar a un comportamiento distinto
        # al de SuperMemo 2.
        self.difficulty = None
        self.hist_correct = 0

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
        self.difficulty += -0.8 + 0.28 * performance + 0.02 * performance ** 2

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