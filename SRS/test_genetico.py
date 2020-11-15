import numpy as np
from Evolutivo.evolutivo import GA

n_bits = 30

# Alfa, Umbral, Phi(5), Psi(5)
var_bits = np.ones(12, dtype=int) * 30
var_lims = np.arange(0, 13) * var_bits
var_min  = np.array(
    [0, 0, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]
)
var_max  = np.array(
    [100, 1, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
)

def main():
    evolutivo_kwargs = {
                'N'                : 400,
                'v_var'            : var_bits,
                'probCrossOver'    : 0.9,
                'probMutation'     : 0.2,
                'f_deco'           : DecoDecimal,
                'f_fitness'        : f_fitness,
                'maxGens'          : 1000,
                'debugLvl'         : 3,
    }

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

if __name__ == "__main__":
    main()