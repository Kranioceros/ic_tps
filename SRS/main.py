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

interrev = 1 * 24 * 3600
nvent = 5
ancho_ventanas = np.exp( np.log(15) / nvent * np.arange(1, nvent+1))
ancho_ventanas *= (24 * 3600)


def main():
    #Cargar las matrices m_t, m_c, m_s y a lens
    m_t = np.load('SRS/data/times.npy')
    m_c = np.load('SRS/data/correct.npy')
    m_s = np.load('SRS/data/seen.npy')
    lens = np.load('SRS/data/len_schedule.npy')
    lens = lens.astype(int)

    # NUMEROS MAGICOS
    N = m_t.shape[0]
    N_parts = 5
    n = 10

    # Calculamos particiones
    len_part = int(N / 5)
    parts = []
    idx = np.arange(0, N)
    np.random.shuffle(idx)
    for i in range(N_parts-1):
        parts.append(idx[i*len_part:(i+1)*len_part])
    parts.append(idx[(N_parts-1)*len_part:])

    # Inicializamos la clase SRGA, que preprocesa los datos si hace falta
    SRGA.init_class(lens, m_t, m_c, m_s, res=1000)

    mejores_fitnesses = []

    for part in tqdm(parts):
        # Definimos la funcion de fitness a utilizar (depende de algunos datos cargados)
        def f_fitness(vars):
            alfa0 = vars[0]
            umbral = vars[1]
            phi = vars[2:7]
            psi = vars[7:]

            srga = SRGA(alfa0, phi, psi, umbral)

            v_apts = np.zeros(n)
            scheds = np.random.choice(part, size=n)

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
                    'maxGens'          : 10,
                    'debugLvl'         : 90,
        }

        #Evolucionamos
        ga = GA(**evolutivo_kwargs)
        ga.Evolve(brecha=0)
        # Guardamos datos
        mejores_fitnesses.append(ga.bestFitness)


    # Imprimimos los mejores fitnesses
    print(f"Mejores fitnesses: {mejores_fitnesses}")
    # Calcular media, varianza, boxplot...

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