import numpy as np
from tqdm import tqdm
import apsw

def main():
    # Antes de correr este programa:
    #   * Crear carpeta 'data' dentro de SRS y guardar la base de datos ahi dentro
    #     con nombre 'duolingo.csv'
    #   * Usar create_duolingo_raw.sql para crear la tabla principal
    #   * Luego usar el comando ".import" para cargar el CSV en la tabla (.mode en CSV!)
    #   * Finalmente, usar create_users_lexemes_30 para filtrar los usuarios relevantes

    # Conectamos con la base de datos
    db = apsw.Connection('SRS/data/duolingo.db')

    # Contamos el numero de pares (user_lexeme) en consideracion. Esto determina
    # el numero de schedules reales.
    schedules_count = None
    for (c,) in db.cursor().execute('SELECT count(*) FROM users_lexemes_30'):
        schedules_count = c

    # Buscamos cual es el nro maximo de sesiones para los pares (user_lexeme)
    sessions_max = None
    for (m,) in db.cursor().execute('SELECT max(sessions_count) FROM users_lexemes_30'):
        sessions_max = m

    # m_times:
    #   Matriz con tiempos de cada schedule
    # v_len_schedule:
    #   Numero de sesiones por schedule
    # m_seen:
    #   Matriz con nro. total de lexemes en cada sesion por schedule
    # m_correct:
    #   Matriz con nro. de lexemes recordados correctamente en cada sesion por schedule
    m_times = np.ones((schedules_count, sessions_max), dtype=np.int32) * (-1)
    v_len_schedule = np.zeros(schedules_count)
    m_seen = np.ones((schedules_count, sessions_max), dtype=np.int32) * (-1)
    m_correct = np.ones((schedules_count, sessions_max), dtype=np.int32) * (-1)
    for i, (user_id, lexeme_id, session_count) in tqdm(enumerate(db.cursor().execute(
        'select * from users_lexemes_30'))):
        for j, (t,seen, correct) in enumerate(db.cursor().execute(
            f"""SELECT timestamp, session_seen, session_correct
	        FROM duolingo_raw
	        WHERE user_id = '{user_id}' AND lexeme_id = '{lexeme_id}'
	        ORDER BY timestamp ASC""")):
                m_times[i,j] = t
                m_seen[i, j] = seen
                m_correct[i, j] = correct
        # Hacemos los tiempos con relativos al inicio del schedule
        m_times[i, :session_count] -= m_times[i, 0]
        # Guardamos el largo del schedule
        v_len_schedule[i] = session_count
    
    # Guardamos los datos obtenidos como ndarrays serializados
    np.save('SRS/data/times', m_times)
    np.save('SRS/data/len_schedule', v_len_schedule)
    np.save('SRS/data/seen', m_seen)
    np.save('SRS/data/correct', m_correct)

#si lo queremos separado..
#m_lexemes_ids = []
#m_deltas = np.zeros(schedules_count)
#    for i, (lexeme_id, lex_dificulty) in tqdm(enumerate(db.cursor().execute(
#       'select * from only_dificulty'))):
#       m_lexemes_ids.append(lexeme_id)
#       m_deltas[i] = lex_dificulty

#Con esto vamos a tener una matriz (lista de tuplas en realidad) a la par de las demas, que contienen los lexemes_dificulty de cada palabra 

   # for i, (lexeme_id, lex_dificulty) in tqdm(enumerate(db.cursor().execute(
   #    'select * from only_dificulty'))):
   #    m_lexemes_dificulty.append(lexeme_id,lex_dificulty)

#lexemes = np.genfromtxt("SRS/data/only_dificulty_30.csv", dtype='U25', delimiter=',')
#print(lexemes)
#v_only_dif = lexemes[:,1].astype(np.float)
#v_lexemes_ids = lexemes[:,0]
#print(v_lexemes_ids)

if __name__ == "__main__":
    main()