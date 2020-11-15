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

    # Reordenamos al azar
    idx = np.random.shuffle(np.arange(N, dtype=int))
    m_t = m_t[idx].reshape(N, 352)
    m_c = m_c[idx].reshape(N, 352)
    m_s = m_s[idx].reshape(N, 352)
    lens = lens[idx].reshape(N)

    print(lens)

    # Creamos un SRS uniforme
    uniforme = Uniforme(24 * 3600)

    # Creamos nuestro SRS deluxe, SRGA
    nvent = 5
    phi = np.flip(np.linspace(0.10, 0.60, num=nvent))
    psi = np.flip(-np.linspace(0.10, 0.60, num=nvent))
    srga = SRGA(1.0, phi, psi, nvent, 0.60)

    # Lo probamos contra todos los schedules
    v_apts = np.ones(N) * (-1)
    descartados = 0
    umbral_fitness = 5
    interrev = 1 * 24 * 3600

    for i in tqdm(range(N)):
        l = lens[i]
        if m_t[i,l-1] < interrev:
            descartados += 1
            continue

        v_apts[i] = fitness(m_t[i,:l], m_c[i,:l], m_s[i,:l], srga,
            return_revs=False, alfa=0.5, interrev=interrev)

    v_apts_sorted = np.sort(v_apts[v_apts>0])
    print(f'10 peores: {v_apts_sorted[:10]}')
    print(f'10 mejores: {v_apts_sorted[-10:]}')
    print(f'Descartados: {descartados}')
    print(f'Menores a {umbral_fitness}: {np.sum(v_apts_sorted < umbral_fitness)}')
    print(f'Avg: {np.average(v_apts)}')


if __name__ == "__main__":
    main()