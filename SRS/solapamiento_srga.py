import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FuncFormatter, AutoMinorLocator)

from tqdm import tqdm
from utils import graficar, integral_acumulada, plotLogisticaOpt, simil
from srs import SRS, SRGA

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

    # Relacionados a SRS
    nvent = 5
    phi = np.flip(np.linspace(0.10, 0.20, num=nvent))
    psi = np.flip(-np.linspace(0.20, 0.40, num=nvent))
    a = 0.4
    delta = 1
    umbral = 0.8
    sigma = 30 # en minutos

    # Parametros (usan variables definidas arriba)
    srga_kwargs = {
        'alfa0': a,
        'phi0': phi,
        'psi0': psi,
        'umbral': umbral,
    }

    # Inicizalizamos clase
    SRGA.init_class(lens, m_t, m_c, m_s, m_d, res=res, sigma=sigma) # Sigma en minutos
    srga = SRGA(**srga_kwargs)

    # Cargamos los datos de cada schedule
    # pylint: disable=unsubscriptable-object, unsupported-assignment-operation
    acum_cs = SRGA.m_acum_cs[sched]
    acum_ss = SRGA.m_acum_ss[sched]

    # Graficas
    _fig, axs = plt.subplots(2,1)
    axs[0].set_title('Similitud para un calendario de 10 y 30 revisiones')
    axs[1].set_xlabel('Días')

    #Primer ejemplo
    ult_rev_1 = 10

    #Similitud
    (similitud_1, ts_srga_1) = simil(ts[0:ult_rev_1], sched, srga, k=1/(3600*24))

    graficar(axs[0], ts, cs, ss, ult_rev_1+1, dens=None, accum=None)
    axs[0].vlines(ts_srga_1[1:], ymin=0, ymax=1, color='g')

    print(f"similitud con {ult_rev_1} revisiones: {similitud_1}")


    #Segundo ejemplo
    ult_rev_2 = 30

    #Similitud
    (similitud_2, ts_srga_2) = simil(ts[0:ult_rev_2+1], sched, srga, k=1/(3600*24))

    graficar(axs[1], ts, cs, ss, ult_rev_2+1, dens=None, accum=None)
    axs[1].vlines(ts_srga_2[1:], ymin=0, ymax=1, color='g')

    print(f"similitud con {ult_rev_2} revisiones: {similitud_2}")

    plt.show()

if __name__ == "__main__":
    main()