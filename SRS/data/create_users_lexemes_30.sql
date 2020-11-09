CREATE TABLE users_lexemes_30 AS SELECT user_id, lexeme_id, count(*) AS sessions_count FROM duolingo_raw
	GROUP BY user_id, lexeme_id
	HAVING count(*) >= 30
	ORDER BY count(*) DESC