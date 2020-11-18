import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FuncFormatter,
                               AutoMinorLocator)

from utils import (graficar, PrLogistica, integral_acumulada,
    PrLogisticaOpt, plotLogisticaOpt)
from tqdm import tqdm
from srs import SRGA

def main():
    # Formatting de la grafica
    fig, axs = plt.subplots(2,1)

    #----- Format --------------
    seg_dia = 24 * 60 * 60
    axs[0].set_ylim(0, 1.1)
    axs[0].xaxis.set_major_locator(MultipleLocator(seg_dia))
    axs[0].xaxis.set_major_formatter(FuncFormatter(lambda x,_p: str(int(x/seg_dia))+'d'))

    axs[1].set_ylim(0, 1.1)
    axs[1].xaxis.set_major_locator(MultipleLocator(seg_dia))
    axs[1].xaxis.set_major_formatter(FuncFormatter(lambda x,_p: str(int(x/seg_dia))+'d'))
    #---------------------------

    # Cargamos datos
    m_t = np.load('SRS/data/times.npy')
    m_c = np.load('SRS/data/correct.npy')
    m_s = np.load('SRS/data/seen.npy')
    lens = np.load('SRS/data/len_schedule.npy')
    lens = lens.astype(int)

    nvent = 5

    # Datos de revisiones y parametros de SRS
    ts = np.array([0,3600*3*24+12,3600*6*24+12,3600*11*24])
    cs = np.ones(4) * 15
    ss = np.ones(4) * 15
    #phi = np.ones(5) * 0.01
    #psi = np.ones(5) * (-0.01)
    phi = np.flip(np.linspace(0.10, 0.20, num=nvent))
    psi = np.flip(-np.linspace(0.10, 0.20, num=nvent))

    # PrLogistica(a,d,phi,psi,ts,cs,ss,t,nvent):
    prlogistica_kwargs = {
        'a':     0.4,
        'd':     2,
        'phi': phi,
        'psi': psi,
    }

    #alfa0, phi0, psi0, nvent, ancho_ventanas, umbral, m_3d
    #srga = SRGA(1,phi,psi,nvent,ancho_ventanas, 0.75, m_3d)

    # --- PrLogistiva referencia -------
    # Evaluamos funcion de probabilidad usada en el SRS
    t = np.linspace(0, 3600*24*15, 1000)
    pr_vector = np.zeros(t.size)
    last_idx = 0
    for rev in range(ts.size):
        #print(f'rev: {rev}')
        t_ultima_rev = ts[rev]
        t_siguiente_rev = ts[rev+1] if rev < ts.size - 1 else 3600*24*15
        #print(f't_ultima_rev: {t_ultima_rev / 3600}')
        #print(f't_siguiente_rev: {t_siguiente_rev / 3600}')
        mask = ts <= t_ultima_rev
        mask2 = np.logical_and(t >= t_ultima_rev, t < t_siguiente_rev)
        for i, t_i in enumerate(t[mask2]):
            #print('t_i: ', t_i / 3600 / 24)
            pr_vector[last_idx] = PrLogistica(**prlogistica_kwargs, ts=ts[mask],
                cs=cs[mask], ss=ss[mask], t=t_i-t_ultima_rev, nvent=5)
            last_idx += 1

    #Grafica de probabilidad de referencia
    axs[0].plot(t, pr_vector)
    axs[0].vlines(ts, ymin=0, ymax=1, color='r', linestyle='dashed')

    # --- PrLogistica optimizada -------
    acum_cs = integral_acumulada(t, ts, cs, sigma=0.5*3600)
    acum_ss = integral_acumulada(t, ts, ss, sigma=0.5*3600)

    plotLogisticaOpt(axs[1], **prlogistica_kwargs, ts_rev=ts[:2], acum_cs=acum_cs,
        acum_ss=acum_ss)

    plt.show()

if __name__ == "__main__":
    main()