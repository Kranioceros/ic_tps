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

    #Cargamos los acums ("ASCO")
    m_acum_cs = np.load('SRS/data/acum_c-res1000-sigma30.npy')
    m_acum_ss = np.load('SRS/data/acum_s-res1000-sigma30.npy')

    # NUMEROS MAGICOS
    N = m_t.shape[0]    #Cantidad total de schedules
    n = 10              #Cantidad de schedules para cada individuo

    # Calculamos particiones
    N_parts = 5
    prct_train = .8
    len_part_train = int(N*prct_train / 5)

    parts_train = []
    parts_test = []
    idx = np.arange(0, N)
    np.random.shuffle(idx)

    for i in range(N_parts-1):
        idxs_train = idx[i*len_part_train:(i+1)*len_part_train]
        parts_train.append(idxs_train)
        parts_test.append(list(set(idx)-set(idxs_train)))
    idxs_train = idx[(N_parts-1)*len_part_train:]
    parts_train.append(idxs_train)
    parts_test.append(list(set(idx) - set(idxs_train)))


    # Inicializamos la clase SRGA, que preprocesa los datos si hace falta
    SRGA.init_class(lens, m_t, m_c, m_s, res=1000)

    mejores_fitnesses = []
    part_alfa = []
    part_phi = []
    part_psi = []
    part_umbral = []

    #Usar particiones de entrenamiento
    print("ENTRENAMIENTO....")
    for i, part in tqdm(enumerate(parts_train)):
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

                v_apts[i] = fitness(s, m_t[s,:l], m_acum_cs[s, -1], m_acum_ss[s, -1], srs=srga)
            
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
        ga.Evolve(elitismo=True, brecha=.4)
        
        # Guardamos datos
        bestAggent = ga.bestAggent
        part_alfa.append(bestAggent[0])
        part_umbral.append(bestAggent[1])
        part_phi.append(bestAggent[2:7])
        part_psi.append(bestAggent[7:])
        mejores_fitnesses.append(ga.bestFitness)

        print(f"INFO PARTICION {i+1}:")
        print(f"MEDIA: {np.mean(ga.v_bestFitness)}")
        print(f"STD: {np.std(ga.v_bestFitness)}")
        print(f"MEDIANA: {np.median(ga.v_bestFitness)}")
        print(f"MAX: {np.max(ga.v_bestFitness)}")
        print(f"MIN: {np.min(ga.v_bestFitness)}\n\n")

    # Imprimimos los mejores fitnesses del entrenamiento
    print(f"Mejores fitnesses durante entrenamiento: {mejores_fitnesses}\n\n\n")

    #Particiones de testeo
    part_apts = []
    part_apts_mean = []
    part_apts_std = []

    for i, part in tqdm(enumerate(parts_test)):

        srga = SRGA(part_alfa[i], part_phi[i], part_psi[i], part_umbral[i])

        v_apts = np.zeros(n)
        scheds = np.random.choice(part, size=n)

        for i, s in enumerate(scheds):
            l = lens[s]
            if m_t[s,l-1] < interrev:
                continue

            v_apts[i] = fitness(s, m_t[s,:l], m_acum_cs[s, -1], m_acum_ss[s, -1], srs=srga)
                
        part_apts.append(v_apts)
        part_apts_mean.append(np.mean(v_apts))
        part_apts_std.append(np.std(v_apts))


    #TODO: graficar boxplot...

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