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

# Devuelve (puntuacion, v_revisiones)
def simil(v_t, v_c, v_s, f_srs):
    v_rev = np.array(v_t.size - 1)
    dist = 0
    ultima_rev = 0
    for i in range(v_rev.size):
        r = f_srs(v_t[:i+1], v_c[i], v_s[i])
        v_rev[i] = r
        dist += np.abs(v_t[i] - ultima_rev)
        ultima_rev = r

    return (dist, v_rev)

def srs_uniforme(historia, correctos, total):
    # Tiempo correspondiente a la sesion actual
    t = historia[-1]
    return t + 24 * 60 * 60

# Fitness = 0
# Por cada schedule real (r):
#   simil = SRS(r)
#   fitness += simil * bondad(r)

def bondad(ts, m):
    ti = np.where(ts < ts[-1] - 6*3600)[-1]
    n = -np.log(m)/(ts[-1] - ts[ti])
    return n

def PrLogistica(a,d,phi,psi,c,n, ts, t):
    #ts, desde el primer tiempo, hasta el tiempo en el q estamos parados

    m_matriz = np.zeros((5,len(ts)))

    ts = -ts - t
    m_matriz[0] = ts >= -6*3600
    m_matriz[1] = ts >= -12*3600
    m_matriz[2] = ts >= -24*3600                #esto podriamos evolucionarlo
    m_matriz[3] = ts >= -168*3600
    m_matriz[4] = ts >= -336*3600

    def sigmoid(x):
        return 2*np.reciprocal(1 + np.exp(-x)) 

    acum = 0
    for i,m_i in enumerate(m_matriz):
        sum_de_phi = phi[i]*np.log(1+np.sum(c*m_i))
        sum_de_psi = psi[i]*np.log(1+np.sum(n*m_i))
        resta = sum_de_phi - sum_de_psi
        acum +=  resta 
    
    Pr = sigmoid(a-d+ acum )
    return Pr

if __name__ == "__main__":
    main()