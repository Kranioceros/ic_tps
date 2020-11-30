from srs import SRS

class SM2(SRS):
    # pylint: disable=unsubscriptable-object, unused-variable
    m_t = None
    m_c = None
    m_s = None

    @classmethod
    def init_class(cls, m_t, m_c, m_s):
        # Copiamos datos
        cls.m_t = m_t
        cls.m_c = m_c
        cls.m_s = m_s

    def __init__(self, delta):
        # TODO: Normalizar delta para que sea siempre >= 1.3. Caso contrario,
        # las constantes utilizadas van a llevar a un comportamiento distinto
        # al de SuperMemo 2.
        self.difficulty = delta
        self.hist_correct = 0

    # Devuelve t_revision
    def prox_revision(self, _delta, sched, t_ult_rev):
        dias = 24 * 3600
        prox_rev = t_ult_rev
        correct = SM2.m_c[sched]  
        total = SM2.m_s[sched]
             
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
            prox_rev += 6 * self.difficulty ** (self.hist_correct - 1) * dias
        else:
            prox_rev += 1 * dias

        return prox_rev