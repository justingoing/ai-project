-- init.sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(255) NOT NULL,
    last_name VARCHAR(255) NOT NULL
);

CREATE TABLE click_tests (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    life_number INTEGER NOT NULL,
    rounds_survived INTEGER NOT NULL,
    move_times JSONB NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE voice_tests (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    life_number INTEGER NOT NULL,
    rounds_survived INTEGER NOT NULL,
    move_times JSONB NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);