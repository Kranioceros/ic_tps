import numpy as np
from srs import SRGA,Uniforme
from Evolutivo.evolutivo import GA
from tqdm import tqdm
from utils import fitness

m_t = np.load('SRS/data/times.npy')
m_c = np.load('SRS/data/correct.npy')
m_s = np.load('SRS/data/seen.npy')
lens = np.load('SRS/data/len_schedule.npy')
lens = lens.astype(int)

""" fitness(sched, ts, cs, ss, srs: SRS, return_revs=False, interrev=3600*6,
    alfa=0.4, k=1/(3600*24)):
for cada uniforme t[i]
    creo el uniforme t[i]
    for cada Sched
        calculo el fitness del sched con el uniforme t[i] """


#t = [12,24,36,48,60,72,84,96]
t = np.linspace(12,180,10)
uniformes = []
v_fitness_tes = np.zeros(len(t))
interrev = 1*24*3600
for j,t_aux in tqdm(enumerate(t)):
    v_fitness = np.zeros(m_t.shape[0])
    uniforme = Uniforme(t_aux*3600)
    for i,sched in tqdm(enumerate(m_t)):
        if m_t[i,lens[i]-1] < interrev:
            continue
        v_fitness[i] = fitness(i,m_t[i,:lens[i]],m_c[i,:lens[i]],m_s[i,:lens[i]],uniforme)
    v_fitness_tes[j] = np.mean(v_fitness)

print(v_fitness_tes)

    