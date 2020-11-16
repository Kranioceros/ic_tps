import numpy as np
from evolutivo import GA

n_bits = 30

var_bits = np.ones(10, dtype=int)*n_bits
var_espr = np.array([1,2,3,4,5,6,7,8,9,10], dtype=int)
var_lims = np.array([0, n_bits, n_bits*2, n_bits*3, n_bits*4, n_bits*5, n_bits*6, n_bits*7, n_bits*8, n_bits*9, n_bits*10], dtype=int)
var_min = np.zeros(10, dtype=int)
var_max = np.ones(10, dtype=int)*100

def main():
    evolutivo_kwargs = {
                'N'                : 400,
                'v_var'            : var_bits,
                'probCrossOver'    : 0.9,
                'probMutation'     : 0.2,
                'f_deco'           : DecoDecimal,
                'f_fitness'        : fitness_tst2,
                'maxGens'          : 1000,
                'debugLvl'         : 3,
    }

    ga = GA(**evolutivo_kwargs)
    ga.Evolve(elitismo=True, brecha=.2, convGen=50)
    ga.DebugPopulation()


def decoIdentidad(x):
    return x

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

def numberOfOnes(x):
    n_ones = 0
    for bit in x:
        n_ones += bit
    return n_ones

#En la primera variable (5 bits) suma '1' por cada '1'
#En la segunda variable (3 bits) suma '1' por cada '0'
#En la tercera variable (4 bits) suma '1' por cada '1'
def fitness_tst1(x):
    score = 0

    lim1 = var_bits[0]
    lim2 = var_bits[0] + var_bits[1]
    lim3 = var_bits[0] + var_bits[1] + var_bits[2]

    for i in range(lim1):
        score += x[i]
    for i in range(lim1,lim2):
        score += not(x[i])
    for i in range(lim2, lim3):
        score += x[i]
    return score

def fitness_tst2(x):
    dem = 0
    for i,_esp in enumerate(var_espr):
        dem += np.abs(x[i]-var_espr[i])

    if(dem == 0): dem = 0.01
    return 1 / dem

if __name__ == "__main__":
    main()