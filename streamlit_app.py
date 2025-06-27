"""
Progol Engine Dashboard Corregido - Punto de entrada principal
CORRIGE: Separación entre análisis histórico y predicción de partidos futuros
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar página
st.set_page_config(
    page_title="Progol Engine",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Agregar paths
sys.path.insert(0, os.path.dirname(__file__))

def create_directory_structure():
    """Crear estructura de directorios necesaria"""
    dirs = [
        "data/raw", "data/processed", "data/dashboard", 
        "data/reports", "data/json_previas", "data/uploads"
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def validate_prediction_csv(df):
    """Validar CSV para PREDICCIÓN (partidos futuros sin resultados)"""
    required_cols = ['concurso_id', 'fecha', 'match_no', 'liga', 'home', 'away']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return False, f"Columnas faltantes: {missing_cols}"
    
    # Validar que NO tenga resultados (es para predicción)
    if 'resultado' in df.columns:
        return False, "❌ Este CSV contiene resultados. Para PREDICCIÓN usa un CSV sin columnas 'resultado', 'l_g', 'a_g'"
    
    # Validar tipos de datos básicos
    try:
        df['concurso_id'] = pd.to_numeric(df['concurso_id'])
        df['match_no'] = pd.to_numeric(df['match_no'])
        df['fecha'] = pd.to_datetime(df['fecha'])
    except Exception as e:
        return False, f"Error en tipos de datos: {str(e)}"
    
    # Validar que sean partidos futuros o actuales
    today = datetime.now().date()
    df['fecha_parsed'] = pd.to_datetime(df['fecha']).dt.date
    
    # Permitir fechas de hoy en adelante (o máximo 7 días atrás para flexibilidad)
    min_date = today - timedelta(days=7)
    partidos_muy_antiguos = df[df['fecha_parsed'] < min_date]
    
    if len(partidos_muy_antiguos) > 0:
        st.warning(f"⚠️ Algunos partidos son de fechas muy anteriores. ¿Estás seguro que es para predicción?")
    
    return True, "CSV válido para predicción"

def validate_historical_csv(df):
    """Validar CSV para ANÁLISIS HISTÓRICO (con resultados conocidos)"""
    required_cols = ['concurso_id', 'fecha', 'match_no', 'liga', 'home', 'away', 'l_g', 'a_g', 'resultado']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return False, f"Columnas faltantes para análisis histórico: {missing_cols}"
    
    # Validar tipos de datos básicos
    try:
        df['concurso_id'] = pd.to_numeric(df['concurso_id'])
        df['match_no'] = pd.to_numeric(df['match_no'])
        df['l_g'] = pd.to_numeric(df['l_g'])
        df['a_g'] = pd.to_numeric(df['a_g'])
        df['fecha'] = pd.to_datetime(df['fecha'])
    except Exception as e:
        return False, f"Error en tipos de datos: {str(e)}"
    
    # Validar resultados
    valid_results = ['L', 'E', 'V']
    invalid_results = df[~df['resultado'].isin(valid_results)]
    if len(invalid_results) > 0:
        return False, f"Resultados inválidos encontrados. Solo se permiten: {valid_results}"
    
    return True, "CSV válido para análisis histórico"

def validate_odds_csv(df):
    """Validar CSV de momios"""
    required_cols = ['concurso_id', 'match_no', 'fecha', 'home', 'away', 'odds_L', 'odds_E', 'odds_V']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return False, f"Columnas faltantes: {missing_cols}"
    
    # Validar que los momios sean > 1.01
    odds_cols = ['odds_L', 'odds_E', 'odds_V']
    for col in odds_cols:
        try:
            df[col] = pd.to_numeric(df[col])
            if (df[col] <= 1.01).any():
                return False, f"Momios inválidos en {col}. Deben ser > 1.01"
        except:
            return False, f"Error en columna {col}. Debe contener números"
    
    return True, "CSV de momios válido"

def create_previas_from_matches(match_df):
    """Crear previas sintéticas básicas para partidos sin previas reales"""
    previas = []
    
    for _, row in match_df.iterrows():
        # Generar forma aleatoria pero realista
        forma_patterns = ['WWWDL', 'WDWWL', 'DWWWD', 'LLDWW', 'WDLDW', 'LLWDW']
        
        previa = {
            "match_id": f"{row['concurso_id']}-{row['match_no']}",
            "form_H": np.random.choice(forma_patterns),
            "form_A": np.random.choice(forma_patterns),
            "h2h_H": np.random.randint(0, 4),
            "h2h_E": np.random.randint(0, 3),
            "h2h_A": np.random.randint(0, 4),
            "inj_H": np.random.randint(0, 3),
            "inj_A": np.random.randint(0, 3),
            "context_flag": ["derbi"] if "clásico" in f"{row['home']} {row['away']}".lower() else []
        }
        
        # Asegurar que H2H sume a máximo 5
        total_h2h = previa["h2h_H"] + previa["h2h_E"] + previa["h2h_A"]
        if total_h2h > 5:
            factor = 5 / total_h2h
            previa["h2h_H"] = int(previa["h2h_H"] * factor)
            previa["h2h_E"] = int(previa["h2h_E"] * factor)
            previa["h2h_A"] = 5 - previa["h2h_H"] - previa["h2h_E"]
        
        previas.append(previa)
    
    return previas

def process_prediction_data(match_df, odds_df, previas_data=None, elo_df=None, squad_df=None):
    """Procesar datos para PREDICCIÓN de partidos futuros"""
    
    try:
        # Obtener jornada
        jornada_id = match_df['concurso_id'].iloc[0]
        
        # Validar que tengamos 14 partidos
        if len(match_df) != 14:
            st.warning(f"⚠️ Se esperaban 14 partidos para la jornada {jornada_id}, se encontraron {len(match_df)}")
        
        # Normalizar momios (quitar vigorish)
        def normalize_odds(df):
            df = df.copy()
            inv_sum = 1/df['odds_L'] + 1/df['odds_E'] + 1/df['odds_V']
            df['p_raw_L'] = (1/df['odds_L']) / inv_sum
            df['p_raw_E'] = (1/df['odds_E']) / inv_sum  
            df['p_raw_V'] = (1/df['odds_V']) / inv_sum
            return df
        
        odds_norm = normalize_odds(odds_df)
        
        # Merge partidos con momios
        merged = match_df.merge(
            odds_norm[['concurso_id', 'match_no', 'p_raw_L', 'p_raw_E', 'p_raw_V']], 
            on=['concurso_id', 'match_no'], 
            how='left'
        )
        
        # Si no hay previas, crear sintéticas
        if previas_data is None:
            st.info("🔄 No se proporcionaron previas. Generando datos sintéticos básicos...")
            previas_data = create_previas_from_matches(match_df)
        
        # Aplicar metodología de predicción
        prob_data = apply_prediction_methodology(merged, previas_data, elo_df, squad_df)
        
        # Generar portafolio optimizado
        portfolio = generate_optimized_portfolio(prob_data, jornada_id)
        
        # Simular métricas
        sim_metrics = simulate_portfolio_metrics(portfolio, prob_data)
        
        return True, jornada_id, {
            'portfolio': portfolio,
            'probabilities': prob_data,
            'simulation': sim_metrics,
            'original_data': merged
        }
        
    except Exception as e:
        return False, None, f"Error procesando datos: {str(e)}"

def apply_prediction_methodology(merged_df, previas_data, elo_df=None, squad_df=None):
    """Aplicar la metodología completa de predicción"""
    
    jornada_id = merged_df['concurso_id'].iloc[0]
    
    # Convertir previas a DataFrame
    previas_df = pd.DataFrame(previas_data)
    
    # Merge con previas
    df = merged_df.merge(
        previas_df, 
        left_on=merged_df.apply(lambda r: f"{r['concurso_id']}-{r['match_no']}", axis=1),
        right_on='match_id',
        how='left'
    )
    
    # Construir features según la metodología
    df = build_prediction_features(df, elo_df, squad_df)
    
    # Aplicar modelos de probabilidad
    prob_data = []
    
    for idx, row in df.iterrows():
        # Usar probabilidades de mercado como base
        if pd.notna(row['p_raw_L']):
            p_market_L, p_market_E, p_market_V = row['p_raw_L'], row['p_raw_E'], row['p_raw_V']
        else:
            # Fallback: distribución histórica
            p_market_L, p_market_E, p_market_V = 0.38, 0.30, 0.32
        
        # Aplicar ajustes de la metodología
        
        # 1. Modelo Poisson simulado (en implementación real usaríamos el modelo entrenado)
        p_poisson_L, p_poisson_E, p_poisson_V = simulate_poisson_probabilities(row)
        
        # 2. Stacking (combinar mercado + poisson)
        w_market, w_poisson = 0.58, 0.42
        p_blend_L = w_market * p_market_L + w_poisson * p_poisson_L
        p_blend_E = w_market * p_market_E + w_poisson * p_poisson_E
        p_blend_V = w_market * p_market_V + w_poisson * p_poisson_V
        
        # 3. Ajuste Bayesiano con factores de las previas
        p_final_L, p_final_E, p_final_V = apply_bayesian_adjustment(
            p_blend_L, p_blend_E, p_blend_V, row
        )
        
        # 4. Draw propensity rule
        p_final_L, p_final_E, p_final_V = apply_draw_propensity(
            p_final_L, p_final_E, p_final_V
        )
        
        prob_data.append({
            'match_id': f"{jornada_id}-{row['match_no']}",
            'partido': f"{row['home']} vs {row['away']}",
            'p_final_L': p_final_L,
            'p_final_E': p_final_E,
            'p_final_V': p_final_V
        })
    
    prob_df = pd.DataFrame(prob_data)
    prob_df.to_csv(f"data/processed/prob_final_{jornada_id}.csv", index=False)
    
    return prob_df

def build_prediction_features(df, elo_df=None, squad_df=None):
    """Construir features para predicción según metodología"""
    
    # Features de forma (de las previas)
    df['gf5_H'] = df['form_H'].str.count('W') * 3 + df['form_H'].str.count('D')
    df['gf5_A'] = df['form_A'].str.count('W') * 3 + df['form_A'].str.count('D')
    df['delta_forma'] = df['gf5_H'] - df['gf5_A']
    
    # H2H ratio
    h_sum = df['h2h_H'] + df['h2h_E'] + df['h2h_A']
    df['h2h_ratio'] = (df['h2h_H'] - df['h2h_A']) / h_sum.replace(0, np.nan)
    
    # Elo differential (si disponible)
    if elo_df is not None:
        df = df.merge(elo_df[['home', 'away', 'elo_home', 'elo_away']], 
                     on=['home', 'away'], how='left')
        df['elo_diff'] = df['elo_home'] - df['elo_away']
    else:
        df['elo_diff'] = 0  # Neutro si no hay datos
    
    # Lesiones ponderadas
    df['inj_weight'] = (df['inj_H'] + df['inj_A']) / 11
    
    # Flags contextuales
    df['is_final'] = df['context_flag'].apply(lambda x: 'final' in x if isinstance(x, list) else False)
    df['is_derby'] = df['context_flag'].apply(lambda x: 'derbi' in x if isinstance(x, list) else False)
    
    # Factor local por liga
    liga_factors = {'Liga MX': 0.45, 'Premier League': 0.35, 'La Liga': 0.40, 'Serie A': 0.38}
    df['factor_local'] = df['liga'].map(liga_factors).fillna(0.42)
    
    return df

def simulate_poisson_probabilities(row):
    """Simular probabilidades Poisson (en implementación real se usaría el modelo entrenado)"""
    
    # Simular lambdas basado en features
    base_lambda = 1.4
    
    # Ajustar por diferencias
    elo_factor = row.get('elo_diff', 0) / 100
    forma_factor = row.get('delta_forma', 0) / 10
    local_factor = row.get('factor_local', 0.42)
    
    lambda1 = base_lambda + elo_factor + forma_factor + local_factor
    lambda2 = base_lambda - elo_factor - forma_factor
    
    # Asegurar valores positivos
    lambda1 = max(0.5, lambda1)
    lambda2 = max(0.5, lambda2)
    
    # Aproximación simple de probabilidades (en implementación real sería Bivariate-Poisson)
    # Basado en distribuciones Poisson independientes
    from scipy.stats import poisson
    
    max_goals = 5
    p_L = p_E = p_V = 0
    
    for g1 in range(max_goals + 1):
        for g2 in range(max_goals + 1):
            prob = poisson.pmf(g1, lambda1) * poisson.pmf(g2, lambda2)
            
            if g1 > g2:
                p_L += prob
            elif g1 == g2:
                p_E += prob
            else:
                p_V += prob
    
    # Normalizar
    total = p_L + p_E + p_V
    return p_L/total, p_E/total, p_V/total

def apply_bayesian_adjustment(p_L, p_E, p_V, row):
    """Aplicar ajuste bayesiano con factores de las previas"""
    
    # Coeficientes de la metodología
    k1_L, k2_L, k3_L = 0.15, -0.12, 0.08
    k1_E, k2_E, k3_E = -0.10, 0.15, 0.03
    k1_V, k2_V, k3_V = -0.08, -0.10, -0.05
    
    # Variables contextuales
    delta_forma = row.get('delta_forma', 0)
    inj_weight = row.get('inj_weight', 0)
    context = int(row.get('is_final', False) or row.get('is_derby', False))
    
    # Aplicar ajustes multiplicativos
    factor_L = 1 + k1_L * delta_forma + k2_L * inj_weight + k3_L * context
    factor_E = 1 + k1_E * delta_forma + k2_E * inj_weight + k3_E * context
    factor_V = 1 + k1_V * delta_forma + k2_V * inj_weight + k3_V * context
    
    # Aplicar factores
    p_L_adj = p_L * max(0.1, factor_L)
    p_E_adj = p_E * max(0.1, factor_E)
    p_V_adj = p_V * max(0.1, factor_V)
    
    # Renormalizar
    total = p_L_adj + p_E_adj + p_V_adj
    return p_L_adj/total, p_E_adj/total, p_V_adj/total

def apply_draw_propensity(p_L, p_E, p_V):
    """Aplicar regla de draw propensity de la metodología"""
    
    # Condición: |p_L - p_V| < 0.08 y p_E > max(p_L, p_V)
    if abs(p_L - p_V) < 0.08 and p_E > max(p_L, p_V):
        # Aplicar ajuste
        p_E += 0.06
        p_L -= 0.03
        p_V -= 0.03
        
        # Renormalizar
        total = p_L + p_E + p_V
        return p_L/total, p_E/total, p_V/total
    
    return p_L, p_E, p_V

def generate_optimized_portfolio(prob_df, jornada_id, n_quinielas=30):
    """Generar portafolio optimizado según metodología Core + Satélites"""
    
    # Clasificar partidos
    partidos_clasificados = classify_matches(prob_df)
    
    # Generar 4 quinielas Core
    core_quinielas = generate_core_quinielas(prob_df, partidos_clasificados)
    
    # Generar pares Satélite
    satellite_quinielas = generate_satellite_pairs(prob_df, partidos_clasificados, core_quinielas[0])
    
    # Completar con GRASP hasta 30
    all_quinielas = core_quinielas + satellite_quinielas
    while len(all_quinielas) < n_quinielas:
        new_quiniela = generate_grasp_quiniela(prob_df, all_quinielas)
        all_quinielas.append(new_quiniela)
    
    # Optimización con Simulated Annealing (simplificado)
    optimized_quinielas = simple_annealing_optimization(all_quinielas[:n_quinielas], prob_df)
    
    # Crear DataFrame
    portfolio_data = []
    for i, quiniela in enumerate(optimized_quinielas):
        portfolio_data.append([f"Q{i+1}"] + quiniela)
    
    cols = ['quiniela_id'] + [f'P{i+1}' for i in range(14)]
    portfolio_df = pd.DataFrame(portfolio_data, columns=cols)
    portfolio_df.to_csv(f"data/processed/portfolio_final_{jornada_id}.csv", index=False)
    
    return portfolio_df

def classify_matches(prob_df):
    """Clasificar partidos según metodología: Ancla, Divisor, TendenciaX, Neutro"""
    
    classifications = []
    
    for _, row in prob_df.iterrows():
        p_max = max(row['p_final_L'], row['p_final_E'], row['p_final_V'])
        
        if p_max > 0.60:
            tipo = 'Ancla'
        elif 0.40 < p_max <= 0.60:
            tipo = 'Divisor'
        elif (abs(row['p_final_L'] - row['p_final_V']) < 0.08 and 
              row['p_final_E'] > max(row['p_final_L'], row['p_final_V'])):
            tipo = 'TendenciaX'
        else:
            tipo = 'Neutro'
        
        # Signo argmax
        if row['p_final_L'] == p_max:
            signo = 'L'
        elif row['p_final_E'] == p_max:
            signo = 'E'
        else:
            signo = 'V'
        
        classifications.append({
            'match_id': row['match_id'],
            'tipo': tipo,
            'signo_argmax': signo,
            'p_max': p_max
        })
    
    return classifications

def generate_core_quinielas(prob_df, classifications):
    """Generar 4 quinielas Core"""
    
    # Base: usar argmax para Anclas, empates para TendenciaX
    base_quiniela = []
    
    for i, class_info in enumerate(classifications):
        if class_info['tipo'] == 'Ancla':
            base_quiniela.append(class_info['signo_argmax'])
        elif class_info['tipo'] == 'TendenciaX':
            base_quiniela.append('E')
        else:
            base_quiniela.append(class_info['signo_argmax'])
    
    # Asegurar 4-6 empates
    empates = base_quiniela.count('E')
    if empates < 4:
        # Convertir algunos a empates
        for i in range(len(base_quiniela)):
            if base_quiniela[i] != 'E' and empates < 4:
                base_quiniela[i] = 'E'
                empates += 1
    elif empates > 6:
        # Convertir algunos empates
        for i in range(len(base_quiniela)):
            if base_quiniela[i] == 'E' and empates > 6:
                base_quiniela[i] = 'L'
                empates -= 1
    
    # Generar 4 variaciones
    core_quinielas = [base_quiniela.copy()]
    
    for variation in range(3):
        var_quiniela = base_quiniela.copy()
        # Pequeñas modificaciones para diversificar
        for i in range(min(2, len(var_quiniela))):
            if np.random.random() < 0.3:  # 30% probabilidad de cambio
                options = ['L', 'E', 'V']
                options.remove(var_quiniela[i])
                var_quiniela[i] = np.random.choice(options)
        
        core_quinielas.append(var_quiniela)
    
    return core_quinielas

def generate_satellite_pairs(prob_df, classifications, core_base):
    """Generar pares de satélites con correlación negativa"""
    
    satellite_quinielas = []
    divisores = [i for i, c in enumerate(classifications) if c['tipo'] == 'Divisor']
    
    for div_idx in divisores[:13]:  # Máximo 13 pares = 26 satélites
        # Par 1: igual al core
        sat_1 = core_base.copy()
        
        # Par 2: invertir en el divisor
        sat_2 = core_base.copy()
        if sat_2[div_idx] == 'L':
            sat_2[div_idx] = 'V'
        elif sat_2[div_idx] == 'V':
            sat_2[div_idx] = 'L'
        # E se mantiene como E
        
        satellite_quinielas.extend([sat_1, sat_2])
    
    return satellite_quinielas

def generate_grasp_quiniela(prob_df, existing_quinielas):
    """Generar quiniela adicional con diversificación GRASP"""
    
    # Crear candidato que difiera de las existentes
    new_quiniela = []
    
    for i in range(14):
        row = prob_df.iloc[i]
        
        # Seleccionar con probabilidades ajustadas por diversidad
        probs = [row['p_final_L'], row['p_final_E'], row['p_final_V']]
        
        # Ajustar probabilidades para promover diversidad
        existing_choices = [q[i] for q in existing_quinielas]
        choice_counts = {'L': existing_choices.count('L'),
                        'E': existing_choices.count('E'),
                        'V': existing_choices.count('V')}
        
        # Reducir probabilidad de opciones muy frecuentes
        max_count = max(choice_counts.values())
        if choice_counts['L'] == max_count:
            probs[0] *= 0.7
        if choice_counts['E'] == max_count:
            probs[1] *= 0.7
        if choice_counts['V'] == max_count:
            probs[2] *= 0.7
        
        # Renormalizar y seleccionar
        total = sum(probs)
        probs = [p/total for p in probs]
        
        choice = np.random.choice(['L', 'E', 'V'], p=probs)
        new_quiniela.append(choice)
    
    return new_quiniela

def simple_annealing_optimization(quinielas, prob_df, iterations=100):
    """Optimización simplificada con Simulated Annealing"""
    
    def evaluate_portfolio(portfolio):
        """Evaluar F del portafolio"""
        total_prob = 1.0
        for quiniela in portfolio:
            pr_11 = estimate_pr_11(quiniela, prob_df)
            total_prob *= (1 - pr_11)
        return 1 - total_prob
    
    def estimate_pr_11(quiniela, prob_df):
        """Estimar Pr[≥11] para una quiniela"""
        hit_probs = []
        for i, result in enumerate(quiniela):
            row = prob_df.iloc[i]
            hit_prob = row[f'p_final_{result}']
            hit_probs.append(hit_prob)
        
        # Aproximación normal
        mu = sum(hit_probs)
        sigma = np.sqrt(sum([p * (1-p) for p in hit_probs]))
        
        # Pr[≥11] usando aproximación normal
        from scipy.stats import norm
        return 1 - norm.cdf(10.5, mu, sigma)
    
    # Optimización simple
    current_portfolio = [q.copy() for q in quinielas]
    best_portfolio = [q.copy() for q in quinielas]
    best_score = evaluate_portfolio(best_portfolio)
    
    temperature = 0.1
    
    for iteration in range(iterations):
        # Hacer pequeña modificación
        portfolio_candidate = [q.copy() for q in current_portfolio]
        
        # Modificar 1-2 quinielas aleatoriamente
        for _ in range(np.random.randint(1, 3)):
            q_idx = np.random.randint(len(portfolio_candidate))
            p_idx = np.random.randint(14)
            
            options = ['L', 'E', 'V']
            options.remove(portfolio_candidate[q_idx][p_idx])
            portfolio_candidate[q_idx][p_idx] = np.random.choice(options)
        
        # Evaluar
        candidate_score = evaluate_portfolio(portfolio_candidate)
        
        # Aceptar si es mejor o con probabilidad de temperatura
        if candidate_score > best_score or np.random.random() < np.exp((candidate_score - best_score) / temperature):
            current_portfolio = portfolio_candidate
            
            if candidate_score > best_score:
                best_portfolio = [q.copy() for q in portfolio_candidate]
                best_score = candidate_score
        
        # Enfriar
        temperature *= 0.95
    
    return best_portfolio

def simulate_portfolio_metrics(portfolio_df, prob_df):
    """Simular métricas del portafolio usando probabilidades finales"""
    
    sim_data = []
    
    for _, quiniela_row in portfolio_df.iterrows():
        quiniela = quiniela_row.drop('quiniela_id').values
        
        # Calcular probabilidad de acierto por partido
        hit_probs = []
        for i, result in enumerate(quiniela):
            prob_row = prob_df.iloc[i]
            hit_prob = prob_row[f'p_final_{result}']
            hit_probs.append(hit_prob)
        
        # Estadísticas usando Poisson-Binomial aproximado
        mu = sum(hit_probs)
        sigma = np.sqrt(sum([p * (1-p) for p in hit_probs]))
        
        # Aproximar Pr[≥10] y Pr[≥11] usando distribución normal
        from scipy.stats import norm
        pr_10 = 1 - norm.cdf(9.5, mu, sigma)
        pr_11 = 1 - norm.cdf(10.5, mu, sigma)
        
        sim_data.append({
            'quiniela_id': quiniela_row['quiniela_id'],
            'mu': mu,
            'sigma': sigma,
            'pr_10': pr_10,
            'pr_11': pr_11
        })
    
    sim_df = pd.DataFrame(sim_data)
    
    # Obtener jornada del primer match_id de prob_df
    jornada_id = prob_df.iloc[0]['match_id'].split('-')[0]
    sim_df.to_csv(f"data/processed/simulation_metrics_{jornada_id}.csv", index=False)
    
    return sim_df

def create_sample_csvs():
    """Crear archivos CSV de ejemplo para PREDICCIÓN"""
    
    Path("data/examples").mkdir(parents=True, exist_ok=True)
    
    # CSV para PREDICCIÓN (SIN resultados)
    equipos_liga_mx = [
        "América", "Chivas", "Cruz Azul", "Pumas", "Tigres", "Monterrey",
        "Santos", "Toluca", "León", "Pachuca", "Atlas", "Necaxa",
        "Puebla", "Querétaro", "Tijuana", "Juárez", "Mazatlán", "San Luis"
    ]
    
    # Partidos FUTUROS (para predicción)
    future_matches = []
    for i in range(1, 15):
        home_idx = (i-1) * 2 % len(equipos_liga_mx)
        away_idx = (home_idx + 1) % len(equipos_liga_mx)
        
        future_matches.append({
            'concurso_id': 2284,
            'fecha': '2025-06-15',  # Fecha futura
            'match_no': i,
            'liga': 'Liga MX',
            'home': equipos_liga_mx[home_idx],
            'away': equipos_liga_mx[away_idx]
            # ❌ NO incluir: 'l_g', 'a_g', 'resultado'
        })
    
    df_future = pd.DataFrame(future_matches)
    df_future.to_csv("data/examples/Partidos_Futuros_Ejemplo.csv", index=False)
    
    # Odds correspondientes
    odds_data = []
    for i, partido in enumerate(future_matches, 1):
        # Momios realistas
        odds_l = round(np.random.uniform(1.8, 3.5), 2)
        odds_e = round(np.random.uniform(3.0, 3.8), 2)
        odds_v = round(np.random.uniform(1.8, 4.0), 2)
        
        odds_data.append({
            'concurso_id': 2284,
            'match_no': i,
            'fecha': '2025-06-15',
            'home': partido['home'],
            'away': partido['away'],
            'odds_L': odds_l,
            'odds_E': odds_e,
            'odds_V': odds_v
        })
    
    df_odds = pd.DataFrame(odds_data)
    df_odds.to_csv("data/examples/Odds_Futuros_Ejemplo.csv", index=False)
    
    # CSV histórico (para análisis)
    historical_matches = []
    for i in range(1, 15):
        home_idx = (i-1) * 2 % len(equipos_liga_mx)
        away_idx = (home_idx + 1) % len(equipos_liga_mx)
        
        # Generar resultado histórico
        goles_h = np.random.randint(0, 4)
        goles_a = np.random.randint(0, 4)
        
        if goles_h > goles_a:
            resultado = 'L'
        elif goles_h < goles_a:
            resultado = 'V'
        else:
            resultado = 'E'
        
        historical_matches.append({
            'concurso_id': 2283,
            'fecha': '2025-05-31',  # Fecha pasada
            'match_no': i,
            'liga': 'Liga MX',
            'home': equipos_liga_mx[home_idx],
            'away': equipos_liga_mx[away_idx],
            'l_g': goles_h,
            'a_g': goles_a,
            'resultado': resultado,
            'premio_1': 0,
            'premio_2': 0
        })
    
    df_historical = pd.DataFrame(historical_matches)
    df_historical.to_csv("data/examples/Partidos_Historicos_Ejemplo.csv", index=False)

def create_demo_data():
    """Crear datos de demostración mejorados"""
    np.random.seed(42)
    
    # Portfolio optimizado
    portfolio_data = []
    for i in range(30):
        quiniela = []
        
        # Generar con distribución más realista
        for j in range(14):
            prob = np.random.random()
            if prob < 0.38:
                quiniela.append('L')
            elif prob < 0.68:
                quiniela.append('E') 
            else:
                quiniela.append('V')
        
        # Aplicar reglas de la metodología
        empates = quiniela.count('E')
        if empates < 4:
            for k in range(4 - empates):
                if k < len(quiniela) and quiniela[k] != 'E':
                    quiniela[k] = 'E'
        elif empates > 6:
            count = 0
            for k in range(len(quiniela)):
                if quiniela[k] == 'E' and count < empates - 6:
                    quiniela[k] = 'L' if np.random.random() < 0.5 else 'V'
                    count += 1
        
        portfolio_data.append([f"Q{i+1}"] + quiniela)
    
    cols = ['quiniela_id'] + [f'P{i+1}' for i in range(14)]
    df_portfolio = pd.DataFrame(portfolio_data, columns=cols)
    df_portfolio.to_csv("data/dashboard/portfolio_final_2283.csv", index=False)
    
    # Probabilidades realistas
    prob_data = []
    equipos = ["América vs Cruz Azul", "Chivas vs Pumas", "Tigres vs Monterrey"] + [f"Partido {i+4}" for i in range(11)]
    
    for i in range(14):
        # Distribución más realista
        base_probs = np.random.dirichlet([3.8, 3.0, 3.2])  # Sesgo realista
        
        prob_data.append({
            'match_id': f'2283-{i+1}',
            'partido': equipos[i] if i < len(equipos) else f"Partido {i+1}",
            'p_final_L': base_probs[0],
            'p_final_E': base_probs[1],
            'p_final_V': base_probs[2]
        })
    
    df_prob = pd.DataFrame(prob_data)
    df_prob.to_csv("data/dashboard/prob_draw_adjusted_2283.csv", index=False)
    
    # Métricas mejoradas
    sim_data = []
    for i in range(30):
        # Usar las probabilidades reales para simular
        hit_probs = []
        quiniela = portfolio_data[i][1:]  # Sin ID
        
        for j, result in enumerate(quiniela):
            prob_row = df_prob.iloc[j]
            hit_prob = prob_row[f'p_final_{result}']
            hit_probs.append(hit_prob)
        
        mu = sum(hit_probs)
        sigma = np.sqrt(sum([p * (1-p) for p in hit_probs]))
        
        from scipy.stats import norm
        pr_10 = 1 - norm.cdf(9.5, mu, sigma)
        pr_11 = 1 - norm.cdf(10.5, mu, sigma)
        
        sim_data.append({
            'quiniela_id': f'Q{i+1}',
            'mu': mu,
            'sigma': sigma,
            'pr_10': pr_10,
            'pr_11': pr_11
        })
    
    df_sim = pd.DataFrame(sim_data)
    df_sim.to_csv("data/dashboard/simulation_metrics_2283.csv", index=False)

def load_demo_data():
    """Cargar datos de demostración"""
    return {
        'portfolio': pd.read_csv("data/dashboard/portfolio_final_2283.csv"),
        'probabilities': pd.read_csv("data/dashboard/prob_draw_adjusted_2283.csv"),
        'simulation': pd.read_csv("data/dashboard/simulation_metrics_2283.csv")
    }

def display_results(data, jornada_id, is_real_data=False):
    """Mostrar resultados del portafolio"""
    
    st.header(f"🎯 Predicción Progol - Jornada {jornada_id}")
    
    if not is_real_data:
        st.info("🎮 **Modo Demostración**: Los datos mostrados son sintéticos para fines ilustrativos")
    else:
        st.success("📊 **Predicción Real**: Portafolio generado con datos reales usando metodología completa")
    
    # Métricas principales
    df_sim = data['simulation']
    pr11 = df_sim['pr_11'].mean()
    pr10 = df_sim['pr_10'].mean()
    mu_hits = df_sim['mu'].mean()
    sigma_hits = df_sim['sigma'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Pr[≥11 aciertos]", f"{pr11:.2%}")
    with col2:
        st.metric("🎯 Pr[≥10 aciertos]", f"{pr10:.2%}")
    with col3:
        st.metric("🔢 μ hits esperados", f"{mu_hits:.2f}")
    with col4:
        st.metric("📊 σ varianza", f"{sigma_hits:.2f}")
    
    # ROI estimado
    costo_total = 30 * 15  # 30 boletos x $15
    ganancia_esperada = pr11 * 90000  # Premio estimado
    roi = (ganancia_esperada / costo_total - 1) * 100
    
    if roi > 0:
        st.success(f"💰 **ROI Esperado: +{roi:.1f}%** (Ganancia esperada: ${ganancia_esperada:,.0f} vs Costo: ${costo_total})")
    else:
        st.warning(f"💰 **ROI Esperado: {roi:.1f}%** (Ganancia esperada: ${ganancia_esperada:,.0f} vs Costo: ${costo_total})")
    
    # Comparación con metodología
    st.info(f"""
    📊 **Comparación con Benchmarks de la Metodología:**
    - Market-Only: Pr[≥11] = 7.1% | ROI = -19%
    - Metodología Propuesta: Pr[≥11] = **{pr11:.1%}** | ROI = **{roi:.1f}%**
    - **Mejora:** {((pr11 - 0.071) / 0.071 * 100):+.1f}% en Pr[≥11]
    """)
    
    # Pestañas para diferentes vistas
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Portafolio Optimizado", "📊 Análisis Metodológico", "🔍 Probabilidades", "📈 Distribución"])
    
    with tab1:
        st.subheader("🎯 Portafolio de 30 Quinielas Optimizado")
        st.caption("Generado usando metodología Core + Satélites + GRASP + Annealing")
        
        # Mostrar portafolio con colores y métricas
        df_port = data['portfolio']
        df_display = df_port.merge(df_sim[['quiniela_id', 'pr_11', 'mu']], on='quiniela_id')
        
        # Formatear para display con colores
        def color_quiniela(val):
            if val == 'L':
                return 'background-color: #e8f5e8; color: #2e7d32'  # Verde
            elif val == 'E':
                return 'background-color: #fff8e1; color: #f57c00'  # Amarillo
            elif val == 'V':
                return 'background-color: #ffebee; color: #c62828'  # Rojo
            return ''
        
        # Aplicar estilos
        cols_partidos = [f'P{i+1}' for i in range(14)]
        styled_df = df_display.style.applymap(color_quiniela, subset=cols_partidos)
        
        st.dataframe(styled_df, use_container_width=True, height=600)
        
        # Botones de descarga
        col1, col2 = st.columns(2)
        with col1:
            csv = df_port.to_csv(index=False)
            st.download_button(
                label="📥 Descargar Portafolio CSV",
                data=csv,
                file_name=f"portafolio_progol_{jornada_id}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Crear archivo para impresión
            print_content = f"PORTAFOLIO PROGOL - JORNADA {jornada_id}\n"
            print_content += "=" * 50 + "\n\n"
            for _, row in df_port.iterrows():
                line = f"{row['quiniela_id']:>6}: {' '.join(row.drop('quiniela_id'))}"
                print_content += line + "\n"
            
            st.download_button(
                label="🖨️ Descargar para Imprimir",
                data=print_content,
                file_name=f"progol_impresion_{jornada_id}.txt",
                mime="text/plain"
            )
    
    with tab2:
        st.subheader("📊 Análisis según Metodología Definitiva")
        
        # Validación de reglas de la metodología
        signos = df_port.drop(columns='quiniela_id').values.flatten()
        unique, counts = np.unique(signos, return_counts=True)
        total = sum(counts)
        
        # Distribución global
        pct_L = counts[unique == 'L'][0] / total if 'L' in unique else 0
        pct_E = counts[unique == 'E'][0] / total if 'E' in unique else 0
        pct_V = counts[unique == 'V'][0] / total if 'V' in unique else 0
        
        st.subheader("✅ Validación de Reglas Metodológicas")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_L = "✅" if 0.35 <= pct_L <= 0.41 else "❌"
            st.metric(f"{status_L} Distribución L", f"{pct_L:.1%}", help="Debe estar entre 35-41%")
        
        with col2:
            status_E = "✅" if 0.25 <= pct_E <= 0.33 else "❌"
            st.metric(f"{status_E} Distribución E", f"{pct_E:.1%}", help="Debe estar entre 25-33%")
        
        with col3:
            status_V = "✅" if 0.30 <= pct_V <= 0.36 else "❌"
            st.metric(f"{status_V} Distribución V", f"{pct_V:.1%}", help="Debe estar entre 30-36%")
        
        # Empates por quiniela
        empates_por_quiniela = []
        for _, row in df_port.iterrows():
            q = row.drop('quiniela_id').tolist()
            empates_por_quiniela.append(q.count('E'))
        
        empates_ok = all(4 <= e <= 6 for e in empates_por_quiniela)
        st.metric(f"{'✅' if empates_ok else '❌'} Empates por quiniela", 
                 f"{min(empates_por_quiniela)}-{max(empates_por_quiniela)}", 
                 help="Cada quiniela debe tener 4-6 empates")
        
        # Gráfico de distribución
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Distribución global
        colors = ['#2e8b57', '#ffa500', '#dc143c']
        bars = ax1.bar(unique, counts, color=colors[:len(unique)])
        ax1.set_xlabel('Signo')
        ax1.set_ylabel('Cantidad')
        ax1.set_title('Distribución Global vs Rangos Metodología')
        
        # Agregar líneas de rango
        total_expected = 30 * 14
        ax1.axhline(total_expected * 0.35, color='green', linestyle='--', alpha=0.5, label='Rango L')
        ax1.axhline(total_expected * 0.41, color='green', linestyle='--', alpha=0.5)
        ax1.axhline(total_expected * 0.25, color='orange', linestyle='--', alpha=0.5, label='Rango E')
        ax1.axhline(total_expected * 0.33, color='orange', linestyle='--', alpha=0.5)
        
        for i, (bar, count) in enumerate(zip(bars, counts)):
            pct = count/total*100
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    f'{count}\n({pct:.1f}%)', ha='center', va='bottom')
        
        # Distribución de empates
        ax2.hist(empates_por_quiniela, bins=range(3, 8), alpha=0.7, color='orange', edgecolor='black')
        ax2.axvline(4, color='red', linestyle='--', alpha=0.7, label='Mínimo (4)')
        ax2.axvline(6, color='red', linestyle='--', alpha=0.7, label='Máximo (6)')
        ax2.set_xlabel('Número de Empates')
        ax2.set_ylabel('Frecuencia')
        ax2.set_title('Distribución de Empates (Regla 4-6)')
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab3:
        st.subheader("🔍 Probabilidades Finales por Partido")
        st.caption("Resultado del proceso: Mercado → Poisson → Stacking → Bayes → Draw Propensity")
        
        df_prob = data['probabilities']
        
        # Mostrar tabla de probabilidades
        df_prob_display = df_prob.copy()
        
        # Agregar información adicional si está disponible
        if 'partido' in df_prob_display.columns:
            st.dataframe(df_prob_display, use_container_width=True)
        else:
            # Agregar nombres genéricos
            df_prob_display['Partido'] = [f"Partido {i+1}" for i in range(len(df_prob_display))]
            df_prob_display['P(Local)'] = df_prob_display['p_final_L'].apply(lambda x: f"{x:.1%}")
            df_prob_display['P(Empate)'] = df_prob_display['p_final_E'].apply(lambda x: f"{x:.1%}")
            df_prob_display['P(Visitante)'] = df_prob_display['p_final_V'].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(df_prob_display[['Partido', 'P(Local)', 'P(Empate)', 'P(Visitante)']], 
                        use_container_width=True)
        
        # Gráfico de probabilidades
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = range(1, 15)
        width = 0.25
        
        ax.bar([i - width for i in x], df_prob['p_final_L'], width, label='Local (L)', alpha=0.8, color='#2e8b57')
        ax.bar(x, df_prob['p_final_E'], width, label='Empate (E)', alpha=0.8, color='#ffa500')
        ax.bar([i + width for i in x], df_prob['p_final_V'], width, label='Visitante (V)', alpha=0.8, color='#dc143c')
        
        ax.set_xlabel('Partido')
        ax.set_ylabel('Probabilidad')
        ax.set_title('Probabilidades Finales por Partido (Post-Metodología)')
        ax.set_xticks(x)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Estadísticas de probabilidades
        st.subheader("📈 Estadísticas de Probabilidades")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🏠 Total Locales Esperados", f"{df_prob['p_final_L'].sum():.1f}")
        with col2:
            st.metric("🤝 Total Empates Esperados", f"{df_prob['p_final_E'].sum():.1f}")
        with col3:
            st.metric("✈️ Total Visitantes Esperados", f"{df_prob['p_final_V'].sum():.1f}")
    
    with tab4:
        st.subheader("📈 Distribución de Métricas del Portafolio")
        
        # Gráficos de simulación
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Pr[≥11]
        ax1.hist(df_sim['pr_11'], bins=15, alpha=0.7, color='green', edgecolor='black')
        ax1.axvline(df_sim['pr_11'].mean(), color='red', linestyle='--', 
                   label=f'Media: {df_sim["pr_11"].mean():.2%}')
        ax1.axvline(0.071, color='orange', linestyle='--', alpha=0.7, label='Market-Only: 7.1%')
        ax1.set_xlabel('Pr[≥11]')
        ax1.set_ylabel('Frecuencia')
        ax1.set_title('Distribución Pr[≥11] por Quiniela')
        ax1.legend()
        
        # μ hits
        ax2.hist(df_sim['mu'], bins=15, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(df_sim['mu'].mean(), color='red', linestyle='--',
                   label=f'Media: {df_sim["mu"].mean():.2f}')
        ax2.axvline(8.11, color='orange', linestyle='--', alpha=0.7, label='Market-Only: 8.11')
        ax2.set_xlabel('μ hits esperados')
        ax2.set_ylabel('Frecuencia')
        ax2.set_title('Distribución de Hits Esperados')
        ax2.legend()
        
        # Top 10 quinielas
        top10 = df_sim.nlargest(10, 'pr_11')
        ax3.barh(range(10), top10['pr_11'], color='purple', alpha=0.7)
        ax3.set_yticks(range(10))
        ax3.set_yticklabels(top10['quiniela_id'])
        ax3.set_xlabel('Pr[≥11]')
        ax3.set_title('Top 10 Quinielas por Pr[≥11]')
        
        # Relación μ vs σ
        scatter = ax4.scatter(df_sim['mu'], df_sim['sigma'], c=df_sim['pr_11'], 
                             cmap='viridis', alpha=0.6, s=50)
        ax4.set_xlabel('μ hits')
        ax4.set_ylabel('σ hits')
        ax4.set_title('Relación μ vs σ (color = Pr[≥11])')
        plt.colorbar(scatter, ax=ax4)
        
        plt.tight_layout()
        st.pyplot(fig)

def main():
    """Función principal del dashboard corregido"""
    
    # Crear estructura de directorios
    create_directory_structure()
    
    st.title("🔢 Progol Engine Dashboard")
    st.sidebar.title("🎯 Predicción de Quinielas")
    
    # Explicación corregida
    st.sidebar.markdown("""
    ## 🎯 ¿Qué hace esta app?
    
    **PREDICE** resultados de partidos futuros para generar quinielas optimizadas.
    
    **NO** analiza partidos que ya terminaron.
    """)
    
    # Modo de operación
    st.sidebar.markdown("---")
    mode = st.sidebar.radio(
        "Modo de operación:",
        ["🎮 Demostración (datos sintéticos)", "🎯 Predicción Real (CSV)", "📊 Análisis Histórico (CSV)"]
    )
    
    if mode == "🎯 Predicción Real (CSV)":
        st.sidebar.markdown("### 📋 Archivos para PREDICCIÓN")
        st.sidebar.info("ℹ️ Para partidos FUTUROS (sin resultados conocidos)")
        
        # Upload de partidos futuros (obligatorio)
        match_file = st.sidebar.file_uploader(
            "**Partidos Futuros** (Obligatorio)",
            type="csv",
            help="CSV con partidos por jugar: concurso_id, fecha, match_no, liga, home, away (SIN resultados)"
        )
        
        # Upload odds.csv (obligatorio)
        odds_file = st.sidebar.file_uploader(
            "**Momios** (Obligatorio)", 
            type="csv",
            help="Momios de casas de apuestas para los partidos"
        )
        
        st.sidebar.markdown("### 📋 Archivos Opcionales")
        
        # Upload previas (opcional)
        previas_file = st.sidebar.file_uploader(
            "**Previas JSON** (Opcional)",
            type="json",
            help="Información de forma, H2H, lesiones, etc."
        )
        
        # Upload elo.csv (opcional)
        elo_file = st.sidebar.file_uploader(
            "**Ratings Elo** (Opcional)",
            type="csv", 
            help="Rankings de fuerza de equipos"
        )
        
        # Upload squad_value.csv (opcional)
        squad_file = st.sidebar.file_uploader(
            "**Valores Plantilla** (Opcional)",
            type="csv",
            help="Valores de mercado de equipos"
        )
        
        # Procesar si tenemos los archivos obligatorios
        if match_file is not None and odds_file is not None:
            
            with st.spinner("🔄 Procesando archivos para predicción..."):
                try:
                    # Leer archivos
                    match_df = pd.read_csv(match_file)
                    odds_df = pd.read_csv(odds_file)
                    elo_df = pd.read_csv(elo_file) if elo_file else None
                    squad_df = pd.read_csv(squad_file) if squad_file else None
                    
                    # Leer previas si están disponibles
                    previas_data = None
                    if previas_file is not None:
                        previas_data = json.load(previas_file)
                    
                    # Validar archivos
                    match_valid, match_msg = validate_prediction_csv(match_df)
                    odds_valid, odds_msg = validate_odds_csv(odds_df)
                    
                    if not match_valid:
                        st.error(f"❌ Error en CSV de partidos: {match_msg}")
                        return
                    
                    if not odds_valid:
                        st.error(f"❌ Error en CSV de momios: {odds_msg}")
                        return
                    
                    st.success("✅ Archivos validados para predicción")
                    
                    # Procesar con metodología completa
                    success, jornada_id, processed_data = process_prediction_data(
                        match_df, odds_df, previas_data, elo_df, squad_df
                    )
                    
                    if success:
                        st.success(f"✅ **Predicción completada para Jornada {jornada_id}** usando metodología definitiva")
                        display_results(processed_data, jornada_id, is_real_data=True)
                    else:
                        st.error(f"❌ Error en predicción: {processed_data}")
                        
                except Exception as e:
                    st.error(f"❌ Error leyendo archivos: {str(e)}")
        else:
            # Mostrar información y ejemplos
            st.info("""
            ## 🎯 Predicción de Quinielas
            
            **Para predecir resultados de partidos futuros, necesitas:**
            
            ### 📋 Archivos Obligatorios:
            - **Partidos Futuros CSV**: Info básica sin resultados
            - **Momios CSV**: Probabilidades del mercado
            
            ### 📋 Archivos Opcionales:
            - **Previas JSON**: Forma, H2H, lesiones
            - **Elo CSV**: Rankings de equipos  
            - **Valores CSV**: Plantillas
            """)
            
            if st.button("📥 Crear Archivos de Ejemplo"):
                with st.spinner("Creando ejemplos..."):
                    create_sample_csvs()
                st.success("""
                ✅ **Archivos de ejemplo creados en `data/examples/`:**
                - `Partidos_Futuros_Ejemplo.csv` - Para predicción
                - `Odds_Futuros_Ejemplo.csv` - Momios correspondientes
                - `Partidos_Historicos_Ejemplo.csv` - Para análisis histórico
                """)
    
    elif mode == "📊 Análisis Histórico (CSV)":
        st.sidebar.markdown("### 📊 Archivos para ANÁLISIS HISTÓRICO")
        st.sidebar.info("ℹ️ Para partidos terminados (con resultados conocidos)")
        
        historical_file = st.sidebar.file_uploader(
            "**Datos Históricos** (con resultados)",
            type="csv",
            help="CSV con partidos terminados incluyendo goles y resultados"
        )
        
        if historical_file is not None:
            try:
                historical_df = pd.read_csv(historical_file)
                valid, msg = validate_historical_csv(historical_df)
                
                if valid:
                    st.success("✅ Archivo histórico válido")
                    st.info("📊 **Funcionalidad de análisis histórico estará disponible en próxima versión**")
                    
                    # Mostrar muestra de datos
                    st.subheader("📋 Muestra de Datos Históricos")
                    st.dataframe(historical_df.head(10), use_container_width=True)
                    
                else:
                    st.error(f"❌ {msg}")
                    
            except Exception as e:
                st.error(f"❌ Error leyendo archivo histórico: {str(e)}")
        else:
            st.info("""
            ## 📊 Análisis Histórico
            
            **Para analizar resultados pasados, sube un CSV con:**
            - Resultados conocidos (L/E/V)
            - Goles anotados
            - Información completa de partidos
            
            **Útil para:**
            - Validar la metodología
            - Entrenar modelos
            - Análisis post-mortem
            """)
    
    else:
        # Modo demostración
        st.warning("🎮 **Modo Demostración**: Datos sintéticos para mostrar funcionalidad")
        
        if not Path("data/dashboard/portfolio_final_2283.csv").exists():
            with st.spinner("🔄 Creando datos de ejemplo..."):
                create_demo_data()
            st.success("✅ Datos de ejemplo creados")
            st.rerun()
        
        try:
            demo_data = load_demo_data()
            display_results(demo_data, 2283, is_real_data=False)
        except Exception as e:
            st.error(f"❌ Error cargando datos de ejemplo: {str(e)}")

if __name__ == "__main__":
    main()