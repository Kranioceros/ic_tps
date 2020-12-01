import numpy as np
from Evolutivo.evolutivo import GA
from srs import SM2
from tqdm import tqdm
from utils import fitness

import cProfile

n_bits = 30
n_vars = 3

# Alfa, Beta, Gamma
var_bits = np.ones(n_vars, dtype=int) * 30
var_lims = np.zeros(n_vars+1, dtype=int)
var_lims[:-1] = np.arange(0, n_vars) * var_bits
var_lims[-1] = int(n_bits*n_vars)
var_min  = np.array(
    [-1, -1, -1]
)
var_max  = np.array(
    [1, 1, 1]
)

interrev = 1 * 24 * 3600
nvent = 5
ancho_ventanas = np.exp( np.log(15) / nvent * np.arange(1, nvent+1))
ancho_ventanas *= (24 * 3600)


def main():
    #Cargar las matrices m_t, m_c, m_s y a lens
    m_t = np.load('SRS/data/times.npy')
    m_c = np.load('SRS/data/correct.npy')
    m_s = np.load('SRS/data/seen.npy')
    m_d = np.load('SRS/data/lexemes_dificulty.npy')
    lens = np.load('SRS/data/len_schedule.npy')
    lens = lens.astype(int)


    # Cargamos unos pocos para que corra rapido 
    N = m_t.shape[0]
    n = 50
    rand_idx = np.arange(N)
    np.random.shuffle(rand_idx)
    m_t = m_t[rand_idx]
    m_c = m_c[rand_idx]
    m_s = m_s[rand_idx]
    lens = lens[rand_idx]

    # Inicializamos la clase SRGA, que preprocesa los datos si hace falta
    SM2.init_class(lens, m_t, m_c, m_s, m_d)

    # Definimos la funcion de fitness a utilizar (depende de algunos datos cargados)
    def f_fitness(vars):
        alfa = vars[0]
        beta = vars[1]
        gamma = vars[2]

        srga = SM2(alfa, beta, gamma)

        v_apts = np.zeros(n)
        scheds = np.random.choice(np.arange(0, N), size=n)

        for i, s in enumerate(scheds):
            l = lens[s]
            if m_t[s,l-1] < interrev:
                continue

            v_apts[i] = fitness(s, m_t[s,:l], m_c[s,:l], m_s[s,:l], srs=srga)
        
        return np.average(v_apts)
    
    # Definimos parametros a usar en el evolutivo
    evolutivo_kwargs = {
                'N'                : 20,
                'v_var'            : var_bits,
                'probCrossOver'    : 0.9,
                'probMutation'     : 0.2,
                'f_deco'           : DecoDecimal,
                'f_fitness'        : f_fitness,
                'maxGens'          : 100,
                'debugLvl'         : 3,
    }

    #Evolucionamos
    ga = GA(**evolutivo_kwargs)
    ga.Evolve(elitismo=True, brecha=0.4, convGen=100)
    ga.DebugPopulation()

#Decodificador binario-decimal 
# a y b son los limites inferior y superior para cada variable
def DecoDecimal(v, a=var_min, b=var_max):
    vs = []
    for i in range(len(var_lims)-1):
        vs.append(v[var_lims[i]:var_lims[i+1]])

    xs = []
    xs = np.zeros(a.size)

    for (i,vi) in enumerate(vs):
        k = len(vi)
        d = sum(2**(k-np.array(range(1,k+1)))*vi)
        xs[i] = a[i] + (d*((b[i]-a[i])/((2**k)-1)))

    return xs


if __name__ == "__main__":
    main()
    #cProfile.run("main()")