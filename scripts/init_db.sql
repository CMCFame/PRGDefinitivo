-- Progol Engine Database Initialization Script
-- Version 1.0

-- Create schemas
CREATE SCHEMA IF NOT EXISTS progol;
CREATE SCHEMA IF NOT EXISTS airflow;

-- Set default schema
SET search_path TO progol, public;

-- Create tables for portfolio management
CREATE TABLE IF NOT EXISTS portfolios (
    id SERIAL PRIMARY KEY,
    jornada_id INTEGER NOT NULL,
    fecha_generacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hash_portfolio VARCHAR(40) UNIQUE NOT NULL,
    pr_11 DECIMAL(5,4),
    pr_10 DECIMAL(5,4),
    mu_hits DECIMAL(4,2),
    sigma_hits DECIMAL(4,2),
    roi_esperado DECIMAL(6,4),
    n_quinielas INTEGER DEFAULT 30,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for faster queries
CREATE INDEX idx_portfolios_jornada ON portfolios(jornada_id);
CREATE INDEX idx_portfolios_fecha ON portfolios(fecha_generacion);
CREATE INDEX idx_portfolios_pr11 ON portfolios(pr_11 DESC);

-- Create table for individual quinielas
CREATE TABLE IF NOT EXISTS quinielas (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER REFERENCES portfolios(id) ON DELETE CASCADE,
    quiniela_id VARCHAR(20) NOT NULL,
    quiniela_string VARCHAR(14) NOT NULL, -- 14 caracteres L/E/V
    pr_11 DECIMAL(5,4),
    mu_hits DECIMAL(4,2),
    sigma_hits DECIMAL(4,2),
    tipo VARCHAR(20), -- Core, Satellite, GRASP
    metadata JSONB,
    UNIQUE(portfolio_id, quiniela_id)
);

-- Create index for quinielas
CREATE INDEX idx_quinielas_portfolio ON quinielas(portfolio_id);
CREATE INDEX idx_quinielas_tipo ON quinielas(tipo);

-- Create table for match results (historical)
CREATE TABLE IF NOT EXISTS resultados_historicos (
    id SERIAL PRIMARY KEY,
    concurso_id INTEGER NOT NULL,
    fecha DATE NOT NULL,
    match_no INTEGER NOT NULL,
    liga VARCHAR(50),
    home VARCHAR(100),
    away VARCHAR(100),
    goles_home INTEGER,
    goles_away INTEGER,
    resultado CHAR(1) CHECK (resultado IN ('L', 'E', 'V')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(concurso_id, match_no)
);

-- Create index for historical results
CREATE INDEX idx_resultados_concurso ON resultados_historicos(concurso_id);
CREATE INDEX idx_resultados_fecha ON resultados_historicos(fecha);
CREATE INDEX idx_resultados_equipos ON resultados_historicos(home, away);

-- Create table for model performance tracking
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    fecha DATE NOT NULL,
    jornada_id INTEGER NOT NULL,
    modelo VARCHAR(50) NOT NULL, -- 'poisson', 'bayes', 'final'
    log_loss DECIMAL(6,4),
    brier_score DECIMAL(6,4),
    accuracy DECIMAL(5,4),
    recall_empates DECIMAL(5,4),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for model performance
CREATE INDEX idx_model_fecha ON model_performance(fecha DESC);
CREATE INDEX idx_model_tipo ON model_performance(modelo);

-- Create table for configuration parameters
CREATE TABLE IF NOT EXISTS configuracion (
    id SERIAL PRIMARY KEY,
    clave VARCHAR(100) UNIQUE NOT NULL,
    valor TEXT,
    tipo VARCHAR(20), -- 'string', 'number', 'json'
    descripcion TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert default configuration
INSERT INTO configuracion (clave, valor, tipo, descripcion) VALUES
    ('n_quinielas', '30', 'number', 'Número de quinielas a generar'),
    ('costo_boleto', '15', 'number', 'Costo por boleto en MXN'),
    ('premio_cat2_default', '90000', 'number', 'Premio estimado categoría 2'),
    ('n_montecarlo', '50000', 'number', 'Número de simulaciones Monte Carlo'),
    ('w_raw', '0.58', 'number', 'Peso de probabilidades de mercado'),
    ('w_poisson', '0.42', 'number', 'Peso de modelo Poisson'),
    ('alpha_grasp', '0.15', 'number', 'Parámetro alpha para GRASP'),
    ('t0_annealing', '0.05', 'number', 'Temperatura inicial Simulated Annealing'),
    ('beta_annealing', '0.92', 'number', 'Factor de enfriamiento'),
    ('max_iter_annealing', '2000', 'number', 'Iteraciones máximas sin mejora')
ON CONFLICT (clave) DO NOTHING;

-- Create table for alerts and notifications
CREATE TABLE IF NOT EXISTS alertas (
    id SERIAL PRIMARY KEY,
    tipo VARCHAR(50) NOT NULL, -- 'error', 'warning', 'info', 'success'
    origen VARCHAR(100) NOT NULL, -- 'etl', 'model', 'optimizer', 'dashboard'
    mensaje TEXT NOT NULL,
    detalles JSONB,
    leida BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for alerts
CREATE INDEX idx_alertas_tipo ON alertas(tipo);
CREATE INDEX idx_alertas_fecha ON alertas(created_at DESC);
CREATE INDEX idx_alertas_leida ON alertas(leida);

-- Create views for quick access
CREATE OR REPLACE VIEW v_ultimo_portfolio AS
SELECT 
    p.*,
    COUNT(q.id) as total_quinielas,
    AVG(q.pr_11) as avg_pr_11_quinielas,
    json_build_object(
        'core', COUNT(q.id) FILTER (WHERE q.tipo = 'Core'),
        'satellite', COUNT(q.id) FILTER (WHERE q.tipo = 'Satellite'),
        'grasp', COUNT(q.id) FILTER (WHERE q.tipo = 'GRASP')
    ) as distribucion_tipos
FROM portfolios p
LEFT JOIN quinielas q ON p.id = q.portfolio_id
WHERE p.fecha_generacion = (SELECT MAX(fecha_generacion) FROM portfolios)
GROUP BY p.id;

CREATE OR REPLACE VIEW v_performance_historico AS
SELECT 
    p.jornada_id,
    p.fecha_generacion::date as fecha,
    p.pr_11 as pr_11_esperado,
    p.roi_esperado,
    mp.accuracy as accuracy_real,
    mp.log_loss,
    CASE 
        WHEN mp.accuracy >= 0.786 THEN '11+ aciertos'
        WHEN mp.accuracy >= 0.714 THEN '10 aciertos'
        WHEN mp.accuracy >= 0.643 THEN '9 aciertos'
        ELSE 'Menos de 9'
    END as categoria_resultado
FROM portfolios p
LEFT JOIN model_performance mp ON p.jornada_id = mp.jornada_id AND mp.modelo = 'final'
ORDER BY p.fecha_generacion DESC;

-- Create functions
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers
CREATE TRIGGER update_portfolios_updated_at
    BEFORE UPDATE ON portfolios
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_configuracion_updated_at
    BEFORE UPDATE ON configuracion
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Grant permissions (adjust as needed)
GRANT ALL PRIVILEGES ON SCHEMA progol TO progol_admin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA progol TO progol_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA progol TO progol_admin;

-- Create read-only user for dashboard
CREATE USER IF NOT EXISTS progol_reader WITH PASSWORD 'readonly_pass';
GRANT CONNECT ON DATABASE progol TO progol_reader;
GRANT USAGE ON SCHEMA progol TO progol_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA progol TO progol_reader;

-- Vacuum and analyze for performance
VACUUM ANALYZE;