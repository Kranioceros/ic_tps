import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FuncFormatter, AutoMinorLocator)

from tqdm import tqdm
from utils import graficar, integral_acumulada, plotLogisticaOpt
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

    # Parametros de plot
    plot_kwargs = {
        'a': a,
        'phi': phi,
        'psi': psi,
        'acum_cs': acum_cs,
        'acum_ss': acum_ss,
    }

    # Parametros de graficar
    graficar_kwargs = {
        'sigma': sigma,
        'res': res,
    }

    # Computamos acumuladas
    ult_rev = 10
    # Calculamos tiempo para cierta revision
    (t, p) = srga.prox_revision(sched, ts[ult_rev])
    print((t / 3600 / 24, p))

    # Inicializamos grafica
    _fig, axs = plt.subplots(3,2)

    graficar(axs[0, 0], ts, cs, ss, ult_rev+1, dens=None, accum=None)
    #axs[1, 0].plot(v_t, acum_cs)
    graficar(axs[1, 0], ts, cs, ss, ult_rev+1, dens=None, accum='CS', **graficar_kwargs)
    graficar(axs[1, 0], ts, cs, ss, ult_rev+1, dens=None, accum='SS', c='g', **graficar_kwargs)
    plotLogisticaOpt(axs[2, 0], **plot_kwargs, d=m_d[0], ts_rev=ts[:ult_rev+1])
    axs[2, 0].scatter(t, p, c='r')
    axs[2, 0].hlines(umbral, 0, 15*24*3600, linestyle='dashed', color='gray')

    # Computamos acumuladas
    ult_rev = 5
    # Calculamos tiempo para cierta revision
    (t, p) = srga.prox_revision(sched, ts[ult_rev])
    print((t / 3600 / 24, p))

    graficar(axs[0, 1], ts, cs, ss, ult_rev+1, dens=None, accum=None)
    graficar(axs[1, 1], ts, cs, ss, ult_rev+1, dens=None, accum='CS', **graficar_kwargs)
    graficar(axs[1, 1], ts, cs, ss, ult_rev+1, dens=None, accum='SS', c='g', **graficar_kwargs)
    plotLogisticaOpt(axs[2, 1], **plot_kwargs, d=m_d[1], ts_rev=ts[:ult_rev+1])
    axs[2, 1].scatter(t, p, c='r')
    axs[2, 1].hlines(umbral, 0, 15*24*3600, linestyle='dashed', color='gray')

    plt.show()

if __name__ == "__main__":
    main()