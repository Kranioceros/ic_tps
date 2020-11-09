CREATE TABLE duolingo_raw (
	p_recall REAL,
	timestamp INTEGER,
	delta INTEGER,
	user_id TEXT,
	learning_language TEXT,
	ui_language TEXT,
	lexeme_id TEXT,
	lexeme_string TEXT,
	history_seen INTEGER,
	history_correct INTEGER,
	session_seen INTEGER,
	session_correct INTEGER,
	PRIMARY KEY (user_id, lexeme_id, timestamp)
);