import numpy as np
from tqdm import tqdm
from pathlib import Path

def main():
    '''lexemes = np.genfromtxt("SRS/data/only_dificulty_30.csv", dtype='U25', delimiter=',')
    v_only_dif = lexemes[:,1].astype(np.float)
    v_lexemes_ids = lexemes[:,0]
    np.save('SRS/data/lexemes_dificulty', v_only_dif)
    np.save('SRS/data/lexemes_ids', v_lexemes_ids)'''

    v_only_dif_path = Path('SRS/data/lexemes_dificulty.npy')
    v_lexemes_ids_path = Path('SRS/data/lexemes_ids.npy')
    if v_only_dif_path.is_file() and v_lexemes_ids_path.is_file():
        print("Cargando datos cacheados...")
        dificultades = np.load(v_only_dif_path)
        ids = np.load(v_lexemes_ids_path)
        
    print("DIFICULTADES:",dificultades)
    print("SEPARADOR-------------------------------------------------------------")
    print("IDS:",ids)
    print("SEPARADOR-------------------------------------------------------------")
    print("Tamaños dif, ids:", len(dificultades), len(ids))

if __name__ == "__main__":
    main()