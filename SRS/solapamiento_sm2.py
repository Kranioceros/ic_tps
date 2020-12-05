import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FuncFormatter, AutoMinorLocator)

from tqdm import tqdm
from utils import graficar, integral_acumulada, plotLogisticaOpt, simil
from srs import SRS, SM2

def main():
 
    # Cargamos datos
    m_t = np.load('SRS/data/times.npy')
    m_c = np.load('SRS/data/correct.npy')
    m_s = np.load('SRS/data/seen.npy')
    m_d = np.load('SRS/data/lexemes_dificulty.npy')
    lens = np.load('SRS/data/len_schedule.npy')
    lens = lens.astype(int)

    ### Datos fijos ###
    sched = 800
    n = lens[sched]
    ts = m_t[sched,:n]
    cs = m_c[sched,:n]
    ss = m_s[sched,:n]

    # Parametros (usan variables definidas arriba)
    sm2_kwargs = {
        'alfa': 0.1,
        'beta': -0.08,
        'gamma': 0.02,
    }

    # Inicizalizamos clase
    SM2.init_class(lens, m_t, m_c, m_s, m_d)
    sm2 = SM2(**sm2_kwargs)

    ult_rev = 10
    # Calculamos tiempo para cierta revision
    (similitud, ts_sm2) = simil(ts[0:ult_rev+1], sched, sm2, k=1/(3600*24))

    # Graficas
    _fig, axs = plt.subplots(1,1)

    #----- Format --------------
    graficar(axs, ts, cs, ss, ult_rev+1, dens=None, accum=None)
    
    axs.set_title('Calendario real y recomendaciones de SM2')
    axs.set_xlabel('Días')
    axs.vlines(ts_sm2[1:], ymin=0, ymax=1, color='g')

    print(ts_sm2)

    plt.show()

if __name__ == "__main__":
    main()