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

    N = 100 #m_t.shape[0]
    interrev = 1 * 24 * 3600
    nvent = 5
    ancho_ventanas = np.exp( np.log(15) / nvent * np.arange(1, nvent+1))
    ancho_ventanas *= (24 * 3600)

    #Procesamiento de los c y n enmascarados
    t_aux = np.linspace(0, 3600*24*15, 100)
    m_3d = np.zeros((N, t_aux.size, 2*nvent))
    for s in tqdm(range(m_3d.shape[0])):
        for t in range(t_aux.size):
            ts_aux = m_t[s] - m_t[s][-1] - t_aux[t]
            for v in range(nvent):
                mask = ts_aux >= -ancho_ventanas[v]
                m_3d[s, t, v] = np.log(1+np.sum(m_c[s]*mask))
                m_3d[s, t, v+nvent] = np.log(1+np.sum(m_s[s]*mask))


    # Datos de revisiones y parametros de SRS
    ts_revs = np.array([0,3600*24,3600*48,3600*72, 3600*96])
    phi = np.flip(np.linspace(0.10, 0.60, num=nvent))
    psi = np.flip(-np.linspace(0.10, 0.60, num=nvent))

    # PrLogistica(a,d,phi,psi,c,n, ts, t):
    prlogistica_kwargs = {
        'a':     1,
        'd':     1,
        'phi': phi,
        'psi': psi,
        'm_3d': m_3d,
    }

    #alfa0, phi0, psi0, nvent, ancho_ventanas, umbral, m_3d
    srga = SRGA(1,phi,psi,nvent,ancho_ventanas, 0.75, m_3d)

    # Evaluamos funcion de probabilidad usada en el SRS
    t_aux = np.linspace(0, 3600*24*15, 100)
    pr_vector = np.zeros(t_aux.size)
    for i, t in enumerate(t_aux):
        pr_vector[i] = PrLogistica(**prlogistica_kwargs, sched=0, t_actual=i, nvent=nvent, ancho_ventanas=ancho_ventanas)

    # Evaluamos obtener el valor del dominio en base a la imagen usando
    # biseccion
    (t_p, p) = srga.prox_revision(0, 50)
    
    # Formatting de la grafica
    _fig = plt.figure()
    ax = plt.subplot()

    seg_dia = 24 * 60 * 60
    #ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(MultipleLocator(seg_dia))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_p: str(int(x/seg_dia))+'d'))

    ax.plot(t_aux, pr_vector)

    ax.scatter(t_p, p, color='red')
    revisiones_aux = m_t[0, m_t[0,:]<t_aux[50]]
    ax.scatter(revisiones_aux, np.ones(revisiones_aux.size)*0.5, color='g')

    plt.show()

if __name__ == "__main__":
    main()