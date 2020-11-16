import numpy as np
from Evolutivo.evolutivo import GA
from srs import SRGA
from tqdm import tqdm
from utils import fitness

import cProfile

n_bits = 30

# Alfa, Umbral, Phi(5), Psi(5)
var_bits = np.ones(12, dtype=int) * 30
var_lims = np.zeros(13, dtype=int)
var_lims[:-1] = np.arange(0, 12) * var_bits
var_lims[-1] = int(n_bits*12)
var_min  = np.array(
    [0, 0, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]
)
var_max  = np.array(
    [100, 1, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
)

#Cargar las matrices m_t, m_c, m_s y a lens
m_t = np.load('SRS/data/times.npy')
m_c = np.load('SRS/data/correct.npy')
m_s = np.load('SRS/data/seen.npy')
lens = np.load('SRS/data/len_schedule.npy')
lens = lens.astype(int)

N = 100 #m_t.shape[0]
interrev = 1 * 24 * 3600
nvent = 5
ancho_ventanas = np.exp( np.log(15) / nvent * np.arange(1, nvent+1))
ancho_ventanas *= (24 * 3600)


def main():
    evolutivo_kwargs = {
                'N'                : 2,
                'v_var'            : var_bits,
                'probCrossOver'    : 0.9,
                'probMutation'     : 0.2,
                'f_deco'           : DecoDecimal,
                'f_fitness'        : f_fitness,
                'maxGens'          : 1,
                'debugLvl'         : 3,
    }

    #Procesamiento de los c y n enmascarados
    t_aux = np.linspace(0, 3600*24*15, 100)
    m_3d = np.zeros(m_t.size, t_aux.size, 2*nvent)
    for s in range(m_3d.shape[0]):
        for t in range(t_aux.size):
            ts_aux = m_t[s] - m_t[s][-1] - t_aux[t]
            for v in range(nvent):
                mask = ts_aux >= -ancho_ventanas[v]
                m_3d[s, t, 2*v] = np.log(1+np.sum(m_c[s]*mask))
                m_3d[s, t, 2*v+1] = np.log(1+np.sum(m_s[s]*mask))

    #Evolucionamos
    ga = GA(**evolutivo_kwargs)
    ga.Evolve()
    ga.DebugPopulation()
    

#Decodificador binario-decimal 
# a y b son los limites inferior y superior para cada variable
def DecoDecimal(v, a=var_min, b=var_max):
    vs = []
    for i in range(len(var_lims)-1):
        vs.append(v[var_lims[i]:var_lims[i+1]])

    xs = []

    for (i,vi) in enumerate(vs):
        k = len(vi)
        d = sum(2**(k-np.array(range(1,k+1)))*vi)
        xs.append(a[i] + (d*((b[i]-a[i])/((2**k)-1))))

    return xs

def f_fitness(vars):
    alfa0 = vars[0]
    umbral = vars[1]
    phi = vars[2:7]
    psi = vars[7:]

    srga = SRGA(alfa0, phi, psi, nvent, ancho_ventanas, umbral)

    v_apts = np.zeros(N)

    for i in tqdm(range(N)):
        l = lens[i]
        if m_t[i,l-1] < interrev:
            continue

        v_apts[i] = fitness(m_t[i,:l], m_c[i,:l], m_s[i,:l], srga,
            return_revs=False, alfa=0.5, interrev=interrev)
    
    return np.average(v_apts)



if __name__ == "__main__":
    cProfile.run("main()")