import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FuncFormatter,
                               AutoMinorLocator)

from utils import graficar, PrLogistica, biseccion
from tqdm import tqdm
from srs import SRGA

def main():
    # Cargamos datos
    m_t = np.load('SRS/data/times.npy')
    m_c = np.load('SRS/data/correct.npy')
    m_s = np.load('SRS/data/seen.npy')
    lens = np.load('SRS/data/len_schedule.npy')
    lens = lens.astype(int)

    nvent = 5

    # Datos de revisiones y parametros de SRS
    ts = np.array([0,3600*3*24,3600*6*24,3600*11*24])
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
        'nvent': 5,
    }

    #alfa0, phi0, psi0, nvent, ancho_ventanas, umbral, m_3d
    #srga = SRGA(1,phi,psi,nvent,ancho_ventanas, 0.75, m_3d)

    # Evaluamos funcion de probabilidad usada en el SRS
    t = np.linspace(0, 3600*24*15, 1000)
    pr_vector = np.zeros(t.size)
    last_idx = 0
    for rev in range(ts.size):
        print(f'rev: {rev}')
        t_ultima_rev = ts[rev]
        t_siguiente_rev = ts[rev+1] if rev < ts.size - 1 else 3600*24*15
        print(f't_ultima_rev: {t_ultima_rev / 3600}')
        print(f't_siguiente_rev: {t_siguiente_rev / 3600}')
        mask = ts <= t_ultima_rev
        mask2 = np.logical_and(t >= t_ultima_rev, t < t_siguiente_rev)
        for i, t_i in enumerate(t[mask2]):
            print('t_i: ', t_i / 3600 / 24)
            pr_vector[last_idx] = PrLogistica(**prlogistica_kwargs, ts=ts[mask],
                cs=cs[mask], ss=ss[mask], t=t_i-t_ultima_rev)
            last_idx += 1

    # Formatting de la grafica
    _fig = plt.figure()
    ax = plt.subplot()

    seg_dia = 24 * 60 * 60
    ax.set_ylim(0, 1.1)
    ax.xaxis.set_major_locator(MultipleLocator(seg_dia))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_p: str(int(x/seg_dia))+'d'))

    ax.plot(t, pr_vector)

    ax.vlines(ts, ymin=0, ymax=1, color='r', linestyle='dashed')

    # Evaluamos obtener el valor del dominio en base a la imagen usando
    # biseccion
    #(t_p, p) = srga.prox_revision(0, 50)
    
    #ax.scatter(t_p, p, color='red')
    #revisiones_aux = m_t[0, m_t[0,:]<t_aux[50]]
    #ax.scatter(revisiones_aux, np.ones(revisiones_aux.size)*0.5, color='g')

    plt.show()

if __name__ == "__main__":
    main()