import numpy as np
import matplotlib.pyplot as plt

def main():
    m_t = np.load('data/times.npy')
    m_c = np.load('data/correct.npy')
    m_s = np.load('data/seen.npy')
    lens = np.load('data/len_schedule.npy')
    idx = 40

    graficar(m_t[idx], m_c[idx], m_s[idx], int(lens[idx]))
    graficar(m_t[idx+1], m_c[idx+1], m_s[idx+1], int(lens[idx+1]))

    plt.show()



def graficar(ts, cs, ss, n):
    mask_cs = np.logical_and((cs == ss),(cs >= 0))
    mask_is = np.logical_not(mask_cs)

    plt.figure()
    plt.stem(ts[mask_cs], np.ones(len(ts[mask_cs])), 'b', linefmt="b-", markerfmt="bo")
    plt.stem(ts[mask_is], np.ones(len(ts[mask_is])), 'r', linefmt="r-", markerfmt="ro")

def ventanas(mt,mc,ms):
    new_mt = np.ones((mt.shape[0],500), dtype=int)*-1
    new_mc = np.ones((mc.shape[0],500), dtype=int)*-1
    new_ms = np.ones((ms.shape[0],500), dtype=int)*-1

    for i in mt.shape[0]:
        pass
    pass


if __name__ == "__main__":
    main()