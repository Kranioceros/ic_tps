import numpy as np
from srs import SRGA,Uniforme
from Evolutivo.evolutivo import GA
from tqdm import tqdm
from utils import fitness
from SM2 import SM2

m_t = np.load('SRS/data/times.npy')
m_c = np.load('SRS/data/correct.npy')
m_s = np.load('SRS/data/seen.npy')
m_d = np.load('SRS/data/lexemes_dificulty.npy')
lens = np.load('SRS/data/len_schedule.npy')
lens = lens.astype(int)

SM2.init_class(lens, m_t, m_c, m_s, m_d)
interrev = 1*24*3600
v_fitness = np.zeros(m_t.shape[0])
for i,sched in tqdm(enumerate(m_t)):
    sm2 = SM2()
    if m_t[i,lens[i]-1] < interrev:
        continue
    v_fitness[i] = fitness(i,m_t[i,:lens[i]],m_c[i,:lens[i]],m_s[i,:lens[i]],sm2)
mean_fitness = np.mean(v_fitness)

print(mean_fitness)
