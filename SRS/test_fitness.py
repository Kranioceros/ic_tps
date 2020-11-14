import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FuncFormatter, AutoMinorLocator)

from utils import fitness, graficar
from srs import SRS, Uniforme

def main():
    # Cargamos datos
    m_t = np.load('SRS/data/times.npy')
    m_c = np.load('SRS/data/correct.npy')
    m_s = np.load('SRS/data/seen.npy')
    lens = np.load('SRS/data/len_schedule.npy')
    lens = lens.astype(int)

    # Armamos un schedule, parecido a un uniforme
    ts_revs = np.array([0,3600*36,3600*50,3600*72, 3600*84])
    c = np.ones(5) * 4
    n = np.ones(5) * 8
    nvent = 5

    # Creamos un SRS uniforme
    uniforme = Uniforme(24 * 3600)

    (revs, apt) = fitness(ts_revs, c, n, uniforme, return_revs=True)
    print(f'aptitud: {apt}')


if __name__ == "__main__":
    main()