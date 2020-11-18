import numpy as np
import matplotlib.pyplot as plt
from utils import densidad,integral_acumulada,graficar

def main():
    m_t = np.load('SRS/data/times.npy')
    m_c = np.load('SRS/data/correct.npy')
    m_s = np.load('SRS/data/seen.npy')
    lens = np.load('SRS/data/len_schedule.npy')
    lens = lens.astype(int)

    fig, axs = plt.subplots(5,1)
    sched = 3000
    cs = m_c[sched]
    ss = m_s[sched]
    n = lens[sched]

    cant_n = 20

    graficar(axs[0], m_t[sched], cs, ss, cant_n, dens=None, accum=None)
    graficar(axs[1], m_t[sched], cs, ss, cant_n, dens='CS', accum=None, sigma=1*3600)
    graficar(axs[2], m_t[sched], cs, ss, cant_n, dens=None, accum='CS', sigma=1*3600)
    graficar(axs[3], m_t[sched], cs, ss, cant_n, dens='SS', accum=None, sigma=1*3600)
    graficar(axs[4], m_t[sched], cs, ss, cant_n, dens=None, accum='SS', sigma=1*3600)

    plt.show()

if __name__ == "__main__":
    main()