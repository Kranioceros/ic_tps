import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FuncFormatter, AutoMinorLocator)

from tqdm import tqdm
from utils import fitness, graficar
from srs import SRS, Uniforme, SRGA

def main():
    # Cargamos datos
    m_t = np.load('SRS/data/times.npy')
    m_c = np.load('SRS/data/correct.npy')
    m_s = np.load('SRS/data/seen.npy')
    lens = np.load('SRS/data/len_schedule.npy')
    lens = lens.astype(int)

    N = m_t.shape[0]
    n = 10

    # Reordenamos al azar
    idx = np.arange(N, dtype=int)
    np.random.shuffle(idx)
    m_t = m_t[idx][:n]
    m_c = m_c[idx][:n]
    m_s = m_s[idx][:n]
    lens = lens[idx][:n]

    # Creamos un SRS uniforme
    _uniforme = Uniforme(24 * 3600)

    # Datos relacionados a SRGA
    res = 1000
    nvent = 5
    phi = np.flip(np.linspace(0.10, 0.20, num=nvent))
    psi = np.flip(-np.linspace(0.20, 0.40, num=nvent))
    a = 0.4
    #delta = 1
    umbral = 0.9
    sigma = 30 # en minutos

    # Creamos nuestro SRS deluxe, SRGA
    srga_kwargs = {
        'alfa0': a,
        'phi0': phi,
        'psi0': psi,
        'umbral': umbral,
    }

    # Inicializamos clase
    SRGA.init_class(lens, m_t, m_c, m_s, res=res, sigma=sigma) # Sigma en minutos
    srga = SRGA(**srga_kwargs)

    # Lo probamos contra todos los schedules
    v_apts = np.ones(n) * (-1)
    descartados = 0
    umbral_fitness = 5
    interrev = 1 * 24 * 3600

    for i in tqdm(range(n)):
        l = lens[i]
        if m_t[i,l-1] < interrev:
            descartados += 1
            continue

        v_apts[i] = fitness(i, m_t[i,:l], m_c[i,:l], m_s[i,:l], srs=srga)
        #v_apts[i] = fitness(m_t[i,:l], m_c[i,:l], m_s[i,:l], srs=srga,
            #return_revs=False, alfa=0.5, interrev=interrev)

    v_apts_sorted = np.sort(v_apts[v_apts>0])
    print(f'10 peores: {v_apts_sorted[:10]}')
    print(f'10 mejores: {v_apts_sorted[-10:]}')
    print(f'Descartados: {descartados}')
    print(f'Menores a {umbral_fitness}: {np.sum(v_apts_sorted < umbral_fitness)}')
    print(f'Avg: {np.average(v_apts)}')


if __name__ == "__main__":
    main()