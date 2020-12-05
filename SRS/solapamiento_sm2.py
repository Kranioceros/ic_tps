import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FuncFormatter, AutoMinorLocator)

from tqdm import tqdm
from utils import graficar, integral_acumulada, plotLogisticaOpt
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
    res = 1000 # resolucion
    n = lens[sched]
    ts = m_t[sched,:n]
    cs = m_c[sched,:n]
    ss = m_s[sched,:n]

    sigma = 30 # en minutos

    # Parametros (usan variables definidas arriba)
    sm2_kwargs = {
        'alfa': 0.1,
        'beta': -0.08,
        'gamma': 0.02,
    }

    # Inicizalizamos clase
    SM2.init_class(lens, m_t, m_c, m_s, m_d) # Sigma en minutos
    sm2 = SM2(**sm2_kwargs)


    ult_rev = 10
    # Calculamos tiempo para cierta revision
    ts_sm2 = np.zeros(ts.size)
    for i, t in enumerate(ts):
        (ts_sm2[i], _p) = sm2.prox_revision(sched, t)

    # Graficas
    _fig, axs = plt.subplots(2,1)

    #----- Format --------------
    seg_dia = 24 * 60 * 60

    graficar(axs[0], ts, cs, ss, ult_rev+1, dens=None, accum=None)
    
    axs[1].set_ylim(0, 1.1)
    axs[1].xaxis.set_major_locator(MultipleLocator(seg_dia))
    axs[1].xaxis.set_major_formatter(FuncFormatter(lambda x,_p: str(int(x/seg_dia))+'d'))
    axs[1].vlines(ts_sm2, ymin=0, ymax=1, color='g')

    plt.show()

if __name__ == "__main__":
    main()