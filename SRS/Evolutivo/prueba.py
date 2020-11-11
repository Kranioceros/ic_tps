import numpy as np
from evolutivo import GA

bit1 = 30
bit2 = 30
bit3 = 30

def main():
    evolutivo_kwargs = {
                'N'                : 100,
                'v_var'            : (bit1, bit2, bit3),
                'probCrossOver'    : 0.9,
                'probMutation'     : 0.1,
                'f_deco'           : DecoDecimal,
                'f_fitness'        : fitness_tst2,
                'maxGens'          : 1000,
                'debugLvl'         : 3,
    }

    ga = GA(**evolutivo_kwargs)
    ga.Evolve()
    ga.DebugPopulation()


def decoIdentidad(x):
    return x

#Decodificador binario-decimal 
# a y b son los limites inferior y superior para cada variable
def DecoDecimal(v, a=(0,1,10), b=(20,5,15)):
    vs = [v[0:bit1], v[bit1:bit1+bit2], v[bit2+bit1:bit1+bit2+bit3]]
    xs = []

    for (i,vi) in enumerate(vs):
        k = len(vi)
        d = sum(2**(k-np.array(range(1,k+1)))*vi)
        xs.append(a[i] + (d*((b[i]-a[i])/((2**k)-1))))

    return (xs[0], xs[1], xs[2])

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
    for i in range(bit1):
        score += x[i]
    for i in range(bit1,bit1+bit2):
        score += not(x[i])
    for i in range(bit1+bit2,bit1+bit2+bit3):
        score += x[i]
    return score

def fitness_tst2(x):
    #numeros esperados para cada variable
    esp_1 = 2.15
    esp_2 = 3.53
    esp_3 = 14.02

    dem = (np.abs(x[0]-esp_1) + np.abs(x[1]-esp_2) + np.abs(x[2]-esp_3))
    if(dem == 0): dem = 0.01
    return 1 / dem

if __name__ == "__main__":
    main()