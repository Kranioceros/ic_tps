CREATE TABLE users_lexemes_30_with_dificulty as select a.*,b.lex_dificulty from users_lexemes_30 as a inner join lexemes_dificulty as b on a.lexeme_id = b.lexeme_id