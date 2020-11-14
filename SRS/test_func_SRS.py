import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FuncFormatter,
                               AutoMinorLocator)

from utils import graficar, PrLogistica, biseccion

def main():
    # Datos de revisiones y parametros de SRS
    ts_revs = np.array([0,3600*24,3600*48,3600*72, 3600*96])
    c = np.ones(5) * 4
    n = np.ones(5) * 8
    nvent = 5
    phi = np.flip(np.linspace(0.10, 0.60, num=nvent))
    psi = np.flip(-np.linspace(0.10, 0.60, num=nvent))

    # PrLogistica(a,d,phi,psi,c,n, ts, t):
    prlogistica_kwargs = {
        'a':     1,
        'd':     1,
        'phi': phi,
        'psi': psi,
        'c':   c,
        'n':   n,
        'ts':  ts_revs,
    }

    # Evaluamos funcion de probabilidad usada en el SRS
    t_aux = np.linspace(0, 3600*24*15, 100)
    pr_vector = np.zeros(t_aux.size)
    for i, t in enumerate(t_aux):
        pr_vector[i] = PrLogistica(**prlogistica_kwargs, t = t, nvent=nvent)

    # Evaluamos obtener el valor del dominio en base a la imagen usando
    # biseccion
    (t_p, p) = biseccion(0.80, 0.1, t_aux, lambda t: PrLogistica(**prlogistica_kwargs, t=t, nvent=nvent), max_iter=10)
    print(f't: {t_p}, p: {p}')
    
    # Formatting de la grafica
    _fig = plt.figure()
    ax = plt.subplot()

    seg_dia = 24 * 60 * 60
    #ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(MultipleLocator(seg_dia))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_p: str(int(x/seg_dia))+'d'))

    ax.plot(t_aux, pr_vector)

    ax.scatter(t_p, p, color='red')

    plt.show()

if __name__ == "__main__":
    main()