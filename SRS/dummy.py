import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FuncFormatter,
                               AutoMinorLocator)

def main():
    m_t = np.load('SRS/data/times.npy')
    m_c = np.load('SRS/data/correct.npy')
    m_s = np.load('SRS/data/seen.npy')
    lens = np.load('SRS/data/len_schedule.npy')

    _fig, axs = plt.subplots(2, 2)

    idx = np.random.choice(np.arange(1000, 1200), size=2)

    # Aprox. el 95% de los valores de una distribucion normal se encuentran en
    # el intervalo [u-2sd; u+2sd]. El 95% del area de cada revision se encuentra
    # en un intervalo de ancho `horas`
    horas = 6
    sd = (horas * 60 * 60) / 2

    ## Sesiones de estudio
    graficar(axs[0, 0], m_t[idx[0]], m_c[idx[0]], m_s[idx[0]], int(lens[idx[0]]))
    graficar(axs[1, 0], m_t[idx[1]], m_c[idx[1]], m_s[idx[1]], int(lens[idx[1]]))
    ## Densidad de estudio
    graficar(axs[0, 1], m_t[idx[0]], m_c[idx[0]], m_s[idx[0]], int(lens[idx[0]]),
        densidad = True, sigma = sd)
    graficar(axs[1, 1], m_t[idx[1]], m_c[idx[1]], m_s[idx[1]], int(lens[idx[1]]),
        densidad = True, sigma = sd)

    #xs = np.arange(-50, 50, step=0.01)
    #v_t = np.array([0])

    #plt.plot(xs, densidad_estudio(xs, v_t, sigma=5))

    plt.show()



def graficar(ax, ts, cs, ss, n, densidad=False, sigma=1.0):
    mask_cs = np.logical_and((cs == ss),(cs >= 0))
    mask_is = np.logical_and(np.logical_not(mask_cs),(cs >= 0))

    # Configuracion grafica
    seg_dia = 24 * 60 * 60
    max_dia = 15
    ticks = np.arange(0, max_dia*seg_dia, step=seg_dia)
    ax.xaxis.set_major_locator(MultipleLocator(seg_dia))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_p: str(int(x/seg_dia))+'d'))
    ax.xaxis.set_minor_locator(MultipleLocator(seg_dia/4))
    ax.set_xlim(right=max_dia*seg_dia)
    ax.vlines(ticks, 0, 1, colors=['grey'], linestyles='dotted')

    if not densidad:
        ax.stem(ts[mask_cs], np.ones(len(ts[mask_cs])), 'b', linefmt="b-",
                markerfmt="bo", basefmt="black")
        ax.stem(ts[mask_is], np.ones(len(ts[mask_is])), 'r', linefmt="r-",
                markerfmt="rx", basefmt="black")
    else:
        print('Laburando...')
        xs = np.linspace(0, max_dia*seg_dia, 50000)
        ys = densidad_estudio(xs, ts, sigma)
        ax.plot(xs, ys)

# `v_t`: tiempos en los que se quiere evaluar la funcion
# `v_mu`: medias correspondientes a cada Gaussiana. Estas medias son
#         iguales al tiempo de cada sesion
# `sigma` es la desviacion estandar. Es unica para todas.
def densidad_estudio(v_t, v_mu, sigma=1):
    gaussianas = np.apply_along_axis(norm_estandar, 1, (v_t[None, :] - v_mu[:, None]) / sigma)
    return np.clip(0, 1, np.sum(gaussianas, 0))

# x puede ser un vector o un escalar
def norm_estandar(x):
    isqrt_2pi = 1 / np.sqrt(2*np.pi)
    return isqrt_2pi * np.exp(-0.5 * x**2)


if __name__ == "__main__":
    main()