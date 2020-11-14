import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FuncFormatter,
                               AutoMinorLocator)

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

def bondad(ts, m):
    ti = np.where(ts < ts[-1] - 6*3600)[-1]
    n = -np.log(m)/(ts[-1] - ts[ti])
    return n

def PrLogistica(a,d,phi,psi,c,n,ts,t,nvent=5):
    ts = ts - ts[-1] - t 

    ancho_ventanas = np.exp( np.log(15) / nvent * np.arange(1, nvent+1))
    ancho_ventanas *= (24 * 3600)
    
    # Calculamos el tamanio de las ventanas
    m_mask = np.zeros((nvent,len(ts)))
    m_mask[0] = ts >= -ancho_ventanas[0]
    for i in range(1, nvent):
        m_mask[i] = ts >= -ancho_ventanas[i]

    def sigmoid(x):
        return np.reciprocal(1 + np.exp(-x))

    acum = 0

    for i, m_i in enumerate(m_mask):
        sum_de_phi = phi[i]*np.log(1+np.sum(c*m_i))
        sum_de_psi = psi[i]*np.log(1+np.sum(n*m_i))
        resta = sum_de_phi - sum_de_psi
        acum += resta 
    
    Pr = sigmoid(a - d + acum)
    return Pr

# Esto funciona para funciones `f` estrictamente decrecientes.
# Devuelve valor mas cercano a x encontrado e imagen de la funcion
# en ese punto
def biseccion(x, tol, v_intervalo, f, max_iter=10):
    N = v_intervalo.size
    idx = int(N / 2)
    y = f(v_intervalo[idx])
    err = x - y
    it = 0
    while it < max_iter and abs(err) > tol:
        if err < 0:
            idx += int((N - idx) / 2)
        else:
            idx -= int(idx / 2)
        it += 1
        y = f(v_intervalo[idx])
        err = x - y
    return (v_intervalo[idx], y)