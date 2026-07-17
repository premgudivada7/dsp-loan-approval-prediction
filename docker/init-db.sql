-- Creates the application database/user (separate from Airflow's metadata DB)
-- and the two tables. The API also create_all()s these at startup; defined here
-- so the ingestion DAG can write stats even before the API has booted.
CREATE USER loan WITH PASSWORD 'loan';
CREATE DATABASE loan OWNER loan;

\connect loan

CREATE TABLE IF NOT EXISTS predictions (
    id           SERIAL PRIMARY KEY,
    created_at   TIMESTAMP DEFAULT NOW(),
    source       VARCHAR(20) DEFAULT 'webapp',
    prediction   VARCHAR(10) NOT NULL,
    probability  DOUBLE PRECISION NOT NULL,
    cibil_score  INTEGER,
    loan_amount  DOUBLE PRECISION,
    income_annum DOUBLE PRECISION,
    features     JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS ingestion_stats (
    id            SERIAL PRIMARY KEY,
    ingested_at   TIMESTAMP DEFAULT NOW(),
    file_name     VARCHAR(255) NOT NULL,
    total_rows    INTEGER NOT NULL,
    valid_rows    INTEGER NOT NULL,
    invalid_rows  INTEGER NOT NULL,
    invalid_pct   DOUBLE PRECISION NOT NULL,
    criticality   VARCHAR(10) NOT NULL,
    error_summary JSONB NOT NULL,
    report_name   VARCHAR(255)
);

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO loan;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO loan;
