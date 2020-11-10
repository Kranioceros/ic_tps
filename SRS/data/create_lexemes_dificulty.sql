CREATE TABLE lexemes_dificulty AS select lexeme_id, sum(history_seen), sum(history_correct), 
(1-sum(history_correct*1.0)/sum(history_seen)) as lex_dificulty
from duolingo_raw
GROUP by lexeme_id