import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FuncFormatter,
                               AutoMinorLocator)

from srs import SRS

def graficar(ax, ts, cs, ss, n, dens=None, accum=None, sigma=1.0):
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

    if dens is None and accum is None:
        ax.stem(ts[mask_cs], np.ones(len(ts[mask_cs])), 'b', linefmt="b-",
                markerfmt="bo", basefmt="black")
        ax.stem(ts[mask_is], np.ones(len(ts[mask_is])), 'r', linefmt="r-",
                markerfmt="rx", basefmt="black")
    elif dens is not None:
        v_x = cs if dens.lower() == "cs" else ss
        xs = np.linspace(0, max_dia*seg_dia, 50000)
        ys = densidad(xs, ts[:n], v_x[:n], sigma)

        print(f'total revisiones: {np.sum(cs[:n])}')
        print(f'ys_integral: {np.trapz(ys, xs)}')

        ax.plot(xs, sigma*ys) # Fines esteticos
    elif accum is not None:
        v_x = cs if accum.lower() == "cs" else ss
        xs = np.linspace(0, max_dia*seg_dia, 5000)
        ys = integral_acumulada(xs, ts[:n], v_x[:n], sigma)

        ax.plot(xs, ys)


# `v_t`: tiempos en los que se quiere evaluar la funcion
# `v_mu`: medias correspondientes a cada Gaussiana. Estas medias son
#         iguales al tiempo de cada sesion
# `sigma` es la desviacion estandar. Es unica para todas.
def densidad(v_t, v_mu, v_x, sigma=1):  

    gaussianas = np.apply_along_axis(norm_estandar, 1, (v_t[None, :] - v_mu[:, None]) / sigma) / sigma
    gaussianas = gaussianas * v_x[:,None]
    return np.sum(gaussianas, 0)

def integral_acumulada(v_t, v_mu, v_x, sigma=1):
    dens = densidad(v_t, v_mu, v_x, sigma)
    accum = np.zeros(v_t.shape)
    for i,_t in enumerate(v_t):
        accum[i] = np.trapz(dens[:i],v_t[:i])
    return accum

# x puede ser un vector o un escalar
# nos quedamos con la mitad de la derecha, 
# preservando el area=1
def norm_estandar(x):
    isqrt_2pi = 1 / np.sqrt(2*np.pi)

    return 2 * (x>=0) * isqrt_2pi * np.exp(-0.5 * x**2)

# Devuelve (puntuacion, v_revisiones)
def simil(ts, cs, ss, srs: SRS, k=3):
    rs = np.zeros(ts.size)
    rs[0] = 0
    for i in range(rs.size-1):
        rs[i+1] = srs.prox_revision(ts[:i+1], cs[i], ss[i])
    
    #print(f"rs: {rs / 3600}")
    #print(f"ts: {ts / 3600}")
    resta = np.abs(ts - rs)
    dist = np.sum(resta)
    #print(f'resta: {resta / 3600}')
    #print(f'dist: {dist / 3600}')
    return (np.exp(-k*dist/ts.size), rs)

def srs_uniforme(historia, correctos, total):
    # Tiempo correspondiente a la sesion actual
    t = historia[-1]
    return t + 24 * 60 * 60

def bondad(ts, m, alfa=0.4, max_rev = 6*3600):
    ti = np.flatnonzero(ts < ts[-1] - max_rev)[-1]
    #print(f'ti: {ti}')
    #print(f't_n: {ts[-1]}')
    #print(f't_(n-1): {ts[ti]}')
    #print(f't_n - t_(n-1): {ts[-1] - ts[ti]}')
    return (m[-1] * (ts[-1] - ts[ti])) ** alfa

def fitness(ts, cs, ss, srs: SRS, return_revs=False, interrev=3600*6,
    alfa=0.4, k=1/(3600*24)):
    (similitud, v_rev) = simil(ts, cs, ss, srs, k=k)
    #print(f'simil: {similitud}')
    buenitud = bondad(ts, cs / ss, alfa=alfa, max_rev=interrev)
    #print(f'bondad: {buenitud}')
    aptitud = buenitud * similitud
    if return_revs:
       return (v_rev, aptitud) 
    else:
       return aptitud

def sigmoid(x):
    return np.reciprocal(1 + np.exp(-x))

# Recibe ts, cs, ss CON LOS EVENTOS A CONSIDERAR. Dado un t_actual, los eventos
# deben ocurrir previamente a t_actual. t es el tiempo con respecto a la ultima
# revision
def PrLogistica(a,d,phi,psi,ts,cs,ss,t,nvent):
    # Si no hay eventos a considerar
    if ts.size == 0:
        return 1.0

    ts = ts - ts[-1] - t

    # Calculamos el tamanio de las ventanas
    ancho_ventanas = np.exp( np.log(15) / nvent * np.arange(1, nvent+1))
    ancho_ventanas *= (24 * 3600)
    #print(f'ancho_ventanas {ancho_ventanas / 3600 / 24}')

    # Calculamos las mascaras para las revisiones
    m_mask = np.zeros((nvent,len(ts)))
    m_mask[0] = ts >= -ancho_ventanas[0]
    for i in range(1, nvent):
        m_mask[i] = ts >= -ancho_ventanas[i]

    acum = 0
    for i, m_i in enumerate(m_mask):
        sum_de_phi = phi[i]*np.log(1+np.sum(cs*m_i))
        sum_de_psi = psi[i]*np.log(1+np.sum(ss*m_i))
        resta = sum_de_phi - sum_de_psi

        #print('mascara_vent: ', m_i)
        #print('sum_de_phi: ', sum_de_phi)
        #print('sum_de_psi: ', sum_de_psi)
        acum += resta 

    Pr = sigmoid(a - d + acum)

    return Pr

# Calculo de probabilidad por metodo de gaussianas
# Recibe el tiempo de la ultima revision (t_actual)
#  y recorta la funcion continua de probabilidad acumulativa hasta t_actual
# Recibe un tiempo para evaluar dentro del dominio (t)
# estudioAcum es la funcion de probabilidad acumulada en todo su dominio
def PrLogisticaOpt(a,d,phi,psi, t_actual, t, estudioAcum_cs, estudioAcum_ss):

    #Recorta la funcion hasta la ultima revision
    idx_tactual = int(np.ceil(t_actual * estudioAcum_cs.size / (15*24*3600)))

    if(idx_tactual==0):
        idx_tactual+=3

    #Copia de los estudios acumulados
    est_cs = np.array(estudioAcum_cs)
    est_ss = np.array(estudioAcum_ss)

    est_cs[idx_tactual+1:] = est_cs[idx_tactual]
    est_ss[idx_tactual+1:] = est_ss[idx_tactual]

    idx_t = int(np.ceil(t * estudioAcum_cs.size / (15*24*3600)))

    #print(f"idx_tactual: {idx_tactual}")
    #print(f"acum_bueno: {estudioAcum_cs[:10]}")
    #print(f"acum_malo: {est_cs[:10]}")
    #cantidad de ventanas
    nvent = phi.size
    #Tamaño de las ventanas
    ancho_ventanas = np.exp( np.log(15) / nvent * np.arange(1, nvent+1))
    ancho_ventanas *= (24 * 3600)

    ancho_ventanas = np.abs(np.ceil(ancho_ventanas * estudioAcum_cs.size / (15*24*3600))).astype(int)

    #Correctas y totales por ventana
    c = np.zeros(nvent)
    s = np.zeros(nvent)

    #print(f"idx_tactual: {idx_tactual}")
    #print(f"ventanas: {ancho_ventanas}")
    #print(f"idx_t: {idx_t}")
    for v in range(nvent):
        ancho_aux = ancho_ventanas[v] if ancho_ventanas[v]<=idx_t else idx_t
        #print(f"ancho_aux: {ancho_aux}")
        #print(f"c_arg1: {est_cs[idx_t]}")
        #print(f"c_arg2: {est_cs[idx_t-ancho_aux]}")
        c[v] = est_cs[idx_t] - est_cs[idx_t-ancho_aux] 
        s[v] = est_ss[idx_t] - est_ss[idx_t-ancho_aux]

    sum_phi = np.dot(phi, np.log(1+c))
    sum_psi = np.dot(psi, np.log(1+s))

    #print(f"C: {c}")
    #print(f"S: {s}")

    resta = sum_phi - sum_psi        
    #print(f"resta: {resta}")
    return sigmoid(a - d + resta)

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