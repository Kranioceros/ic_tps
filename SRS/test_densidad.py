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

    graficar(axs[0], m_t[sched], cs, ss, n, densidad=False, accum=False, sigma=1.0)

    intacum = integral_acumulada(t, m_t[sched], cs, sigma=1)
    dens = densidad(t,ts,cs,sigma=1)
    print("ACCUM",intacum)
    axs[0].plot(t,dens)
    axs[1].plot(t,intacum)
    plt.show()

if __name__ == "__main__":
    main()