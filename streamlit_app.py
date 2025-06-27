"""
Progol Engine Dashboard Mejorado - Punto de entrada principal
Con funcionalidad de carga de datos CSV
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

def validate_progol_csv(df):
    """Validar estructura del CSV de Progol"""
    required_cols = ['concurso_id', 'fecha', 'match_no', 'liga', 'home', 'away', 'l_g', 'a_g', 'resultado']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return False, f"Columnas faltantes: {missing_cols}"
    
    # Validar tipos de datos básicos
    try:
        df['concurso_id'] = pd.to_numeric(df['concurso_id'])
        df['match_no'] = pd.to_numeric(df['match_no'])
        df['l_g'] = pd.to_numeric(df['l_g'])
        df['a_g'] = pd.to_numeric(df['a_g'])
        df['fecha'] = pd.to_datetime(df['fecha'])
    except:
        return False, "Error en tipos de datos. Revisa que concurso_id, match_no, l_g, a_g sean números y fecha sea válida"
    
    # Validar resultados
    valid_results = ['L', 'E', 'V']
    invalid_results = df[~df['resultado'].isin(valid_results)]
    if len(invalid_results) > 0:
        return False, f"Resultados inválidos encontrados. Solo se permiten: {valid_results}"
    
    return True, "CSV válido"

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

def process_uploaded_data(progol_df, odds_df, elo_df=None, squad_df=None):
    """Procesar datos cargados y ejecutar pipeline básico"""
    
    try:
        # Guardar archivos cargados
        progol_df.to_csv("data/uploads/Progol_uploaded.csv", index=False)
        odds_df.to_csv("data/uploads/odds_uploaded.csv", index=False)
        
        if elo_df is not None:
            elo_df.to_csv("data/uploads/elo_uploaded.csv", index=False)
        if squad_df is not None:
            squad_df.to_csv("data/uploads/squad_uploaded.csv", index=False)
        
        # Obtener jornada más reciente
        latest_jornada = progol_df['concurso_id'].max()
        
        # Filtrar datos de la jornada más reciente
        current_jornada = progol_df[progol_df['concurso_id'] == latest_jornada].copy()
        current_odds = odds_df[odds_df['concurso_id'] == latest_jornada].copy()
        
        # Validar que tengamos 14 partidos
        if len(current_jornada) != 14:
            st.warning(f"⚠️ Se esperaban 14 partidos para la jornada {latest_jornada}, se encontraron {len(current_jornada)}")
        
        # Crear datos procesados básicos
        processed_data = create_processed_data(current_jornada, current_odds, latest_jornada)
        
        return True, latest_jornada, processed_data
        
    except Exception as e:
        return False, None, f"Error procesando datos: {str(e)}"

def create_processed_data(progol_df, odds_df, jornada_id):
    """Crear datos procesados básicos"""
    
    # Normalizar momios (quitar vigorish)
    def normalize_odds(df):
        df = df.copy()
        inv_sum = 1/df['odds_L'] + 1/df['odds_E'] + 1/df['odds_V']
        df['p_raw_L'] = (1/df['odds_L']) / inv_sum
        df['p_raw_E'] = (1/df['odds_E']) / inv_sum  
        df['p_raw_V'] = (1/df['odds_V']) / inv_sum
        return df
    
    odds_norm = normalize_odds(odds_df)
    
    # Merge datos
    merged = progol_df.merge(odds_norm[['home', 'away', 'p_raw_L', 'p_raw_E', 'p_raw_V']], 
                            on=['home', 'away'], how='left')
    
    # Crear probabilidades finales (simuladas por ahora)
    prob_data = []
    for idx, row in merged.iterrows():
        if pd.isna(row['p_raw_L']):
            # Si no hay momios, usar distribución por defecto
            p_L, p_E, p_V = 0.38, 0.30, 0.32
        else:
            # Usar momios con pequeño ajuste aleatorio
            p_L = row['p_raw_L'] * np.random.uniform(0.95, 1.05)
            p_E = row['p_raw_E'] * np.random.uniform(0.95, 1.05)
            p_V = row['p_raw_V'] * np.random.uniform(0.95, 1.05)
            
            # Renormalizar
            total = p_L + p_E + p_V
            p_L, p_E, p_V = p_L/total, p_E/total, p_V/total
        
        prob_data.append({
            'match_id': f'{jornada_id}-{row["match_no"]}',
            'p_final_L': p_L,
            'p_final_E': p_E,
            'p_final_V': p_V
        })
    
    prob_df = pd.DataFrame(prob_data)
    prob_df.to_csv(f"data/processed/prob_final_{jornada_id}.csv", index=False)
    
    # Generar portafolio básico
    portfolio = generate_basic_portfolio(prob_df, jornada_id)
    
    # Simular métricas
    sim_metrics = simulate_portfolio_metrics(portfolio, prob_df)
    
    return {
        'portfolio': portfolio,
        'probabilities': prob_df,
        'simulation': sim_metrics,
        'original_data': merged
    }

def generate_basic_portfolio(prob_df, jornada_id, n_quinielas=30):
    """Generar portafolio básico de quinielas"""
    
    portfolio_data = []
    
    for i in range(n_quinielas):
        quiniela = []
        
        for _, row in prob_df.iterrows():
            # Seleccionar resultado basado en probabilidades
            rand = np.random.random()
            if rand < row['p_final_L']:
                result = 'L'
            elif rand < row['p_final_L'] + row['p_final_E']:
                result = 'E'
            else:
                result = 'V'
            
            quiniela.append(result)
        
        # Ajustar para tener 4-6 empates
        empates = quiniela.count('E')
        if empates < 4:
            # Convertir algunos a empates
            for j in range(4 - empates):
                if j < len(quiniela) and quiniela[j] != 'E':
                    quiniela[j] = 'E'
        elif empates > 6:
            # Convertir algunos empates a otros resultados
            count = 0
            for j in range(len(quiniela)):
                if quiniela[j] == 'E' and count < empates - 6:
                    quiniela[j] = 'L' if np.random.random() < 0.5 else 'V'
                    count += 1
        
        portfolio_data.append([f"Q{i+1}"] + quiniela)
    
    # Crear DataFrame
    cols = ['quiniela_id'] + [f'P{i+1}' for i in range(14)]
    portfolio_df = pd.DataFrame(portfolio_data, columns=cols)
    portfolio_df.to_csv(f"data/processed/portfolio_final_{jornada_id}.csv", index=False)
    
    return portfolio_df

def simulate_portfolio_metrics(portfolio_df, prob_df):
    """Simular métricas del portafolio"""
    
    sim_data = []
    
    for _, quiniela_row in portfolio_df.iterrows():
        quiniela = quiniela_row.drop('quiniela_id').values
        
        # Calcular probabilidad de acierto por partido
        hit_probs = []
        for i, result in enumerate(quiniela):
            prob_row = prob_df.iloc[i]
            hit_prob = prob_row[f'p_final_{result}']
            hit_probs.append(hit_prob)
        
        # Estadísticas
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

def main():
    """Función principal del dashboard"""
    
    # Crear estructura de directorios
    create_directory_structure()
    
    st.title("🔢 Progol Engine Dashboard")
    st.sidebar.title("📂 Gestión de Datos")
    
    # Sidebar para carga de datos
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Cargar Datos CSV")
    
    # Modo de operación
    mode = st.sidebar.radio(
        "Modo de operación:",
        ["🎮 Demostración (datos sintéticos)", "📊 Datos reales (CSV)"]
    )
    
    if mode == "📊 Datos reales (CSV)":
        st.sidebar.markdown("### 📋 Archivos Requeridos")
        
        # Upload Progol.csv (obligatorio)
        progol_file = st.sidebar.file_uploader(
            "**Progol.csv** (Obligatorio)",
            type="csv",
            help="Archivo con resultados históricos de concursos"
        )
        
        # Upload odds.csv (obligatorio)
        odds_file = st.sidebar.file_uploader(
            "**Odds.csv** (Obligatorio)", 
            type="csv",
            help="Archivo con momios de casas de apuestas"
        )
        
        st.sidebar.markdown("### 📋 Archivos Opcionales")
        
        # Upload elo.csv (opcional)
        elo_file = st.sidebar.file_uploader(
            "**Elo.csv** (Opcional)",
            type="csv", 
            help="Ratings ELO de equipos"
        )
        
        # Upload squad_value.csv (opcional)
        squad_file = st.sidebar.file_uploader(
            "**Squad_value.csv** (Opcional)",
            type="csv",
            help="Valores de mercado de plantillas"
        )
        
        # Procesar archivos si se cargan los obligatorios
        if progol_file is not None and odds_file is not None:
            
            with st.spinner("🔄 Procesando archivos..."):
                try:
                    # Leer archivos
                    progol_df = pd.read_csv(progol_file)
                    odds_df = pd.read_csv(odds_file)
                    elo_df = pd.read_csv(elo_file) if elo_file else None
                    squad_df = pd.read_csv(squad_file) if squad_file else None
                    
                    # Validar archivos
                    progol_valid, progol_msg = validate_progol_csv(progol_df)
                    odds_valid, odds_msg = validate_odds_csv(odds_df)
                    
                    if not progol_valid:
                        st.error(f"❌ Error en Progol.csv: {progol_msg}")
                        return
                    
                    if not odds_valid:
                        st.error(f"❌ Error en Odds.csv: {odds_msg}")
                        return
                    
                    st.success("✅ Archivos validados correctamente")
                    
                    # Procesar datos
                    success, jornada_id, processed_data = process_uploaded_data(
                        progol_df, odds_df, elo_df, squad_df
                    )
                    
                    if success:
                        st.success(f"✅ Datos procesados. Jornada detectada: **{jornada_id}**")
                        display_results(processed_data, jornada_id, is_real_data=True)
                    else:
                        st.error(f"❌ Error procesando datos: {processed_data}")
                        
                except Exception as e:
                    st.error(f"❌ Error leyendo archivos: {str(e)}")
        else:
            # Mostrar información sobre archivos requeridos
            st.info("""
            📋 **Para usar datos reales, necesitas cargar:**
            
            **Archivos Obligatorios:**
            - **Progol.csv**: Resultados históricos de concursos
            - **Odds.csv**: Momios de casas de apuestas
            
            **Archivos Opcionales:**
            - **Elo.csv**: Ratings de equipos  
            - **Squad_value.csv**: Valores de plantillas
            
            💡 **Tip**: Puedes descargar archivos de ejemplo en la sección de abajo.
            """)
            
            # Botón para descargar ejemplos
            if st.button("📥 Descargar Archivos CSV de Ejemplo"):
                create_sample_csvs()
                st.success("✅ Archivos de ejemplo creados en `data/examples/`")
    
    else:
        # Modo demostración
        st.warning("⚠️ Ejecutándose en modo demostración con datos sintéticos")
        
        # Crear datos de ejemplo si no existen
        if not Path("data/dashboard/portfolio_final_2283.csv").exists():
            with st.spinner("🔄 Creando datos de ejemplo..."):
                create_demo_data()
            st.success("✅ Datos de ejemplo creados")
            st.rerun()
        
        # Cargar y mostrar datos de ejemplo
        try:
            demo_data = load_demo_data()
            display_results(demo_data, 2283, is_real_data=False)
        except Exception as e:
            st.error(f"❌ Error cargando datos de ejemplo: {str(e)}")

def create_sample_csvs():
    """Crear archivos CSV de ejemplo"""
    
    # Crear directorio de ejemplos
    Path("data/examples").mkdir(parents=True, exist_ok=True)
    
    # Progol.csv de ejemplo (jornada 2284)
    equipos_liga_mx = [
        "América", "Chivas", "Cruz Azul", "Pumas", "Tigres", "Monterrey",
        "Santos", "Toluca", "León", "Pachuca", "Atlas", "Necaxa",
        "Puebla", "Querétaro", "Tijuana", "Juárez", "Mazatlán", "San Luis",
        "FC Barcelona", "Real Madrid", "Atlético Madrid", "Sevilla"
    ]
    
    # Generar datos de ejemplo más realistas
    progol_data = []
    for i in range(1, 15):
        home_idx = (i-1) * 2 % len(equipos_liga_mx)
        away_idx = (home_idx + 1) % len(equipos_liga_mx)
        
        # Generar resultado realista
        rand = np.random.random()
        if rand < 0.38:  # Local
            goles_h = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
            goles_a = np.random.choice([0, 1], p=[0.6, 0.4])
            resultado = 'L'
        elif rand < 0.68:  # Empate  
            goles_h = goles_a = np.random.choice([0, 1, 2], p=[0.2, 0.5, 0.3])
            resultado = 'E'
        else:  # Visitante
            goles_a = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
            goles_h = np.random.choice([0, 1], p=[0.6, 0.4])
            resultado = 'V'
        
        progol_data.append({
            'concurso_id': 2284,
            'fecha': '2025-06-07',
            'match_no': i,
            'liga': 'Liga MX' if i <= 10 else 'La Liga',
            'home': equipos_liga_mx[home_idx],
            'away': equipos_liga_mx[away_idx],
            'l_g': goles_h,
            'a_g': goles_a,
            'resultado': resultado,
            'premio_1': 0,
            'premio_2': 0
        })
    
    df_progol = pd.DataFrame(progol_data)
    df_progol.to_csv("data/examples/Progol_ejemplo.csv", index=False)
    
    # Odds.csv de ejemplo
    odds_data = []
    for i, partido in enumerate(progol_data, 1):
        # Generar momios que reflejen el resultado real
        if partido['resultado'] == 'L':
            odds_l = round(np.random.uniform(1.5, 2.2), 2)
            odds_e = round(np.random.uniform(3.0, 3.8), 2)
            odds_v = round(np.random.uniform(3.5, 5.5), 2)
        elif partido['resultado'] == 'V':
            odds_l = round(np.random.uniform(3.5, 5.5), 2)
            odds_e = round(np.random.uniform(3.0, 3.8), 2)
            odds_v = round(np.random.uniform(1.5, 2.2), 2)
        else:  # Empate
            odds_l = round(np.random.uniform(2.5, 3.2), 2)
            odds_e = round(np.random.uniform(2.8, 3.3), 2)
            odds_v = round(np.random.uniform(2.5, 3.2), 2)
        
        odds_data.append({
            'concurso_id': 2284,
            'match_no': i,
            'fecha': '2025-06-07',
            'home': partido['home'],
            'away': partido['away'],
            'odds_L': odds_l,
            'odds_E': odds_e,
            'odds_V': odds_v
        })
    
    df_odds = pd.DataFrame(odds_data)
    df_odds.to_csv("data/examples/Odds_ejemplo.csv", index=False)
    
    # Elo.csv de ejemplo
    elo_data = []
    for equipo in equipos_liga_mx[:14]:
        elo_data.append({
            'home': equipo,
            'away': 'dummy',  # No usado en este ejemplo
            'fecha': '2025-06-07',
            'elo_home': np.random.randint(1500, 1700),
            'elo_away': np.random.randint(1500, 1700),
            'factor_local': 0.45
        })
    
    df_elo = pd.DataFrame(elo_data)
    df_elo.to_csv("data/examples/Elo_ejemplo.csv", index=False)
    
    # Squad_value.csv de ejemplo
    squad_data = []
    for equipo in equipos_liga_mx[:14]:
        squad_data.append({
            'team': equipo,
            'squad_value': np.random.randint(20, 80) * 1000000,  # 20M - 80M
            'avg_age': round(np.random.uniform(23, 29), 1),
            'internationals': np.random.randint(3, 12)
        })
    
    df_squad = pd.DataFrame(squad_data)
    df_squad.to_csv("data/examples/Squad_value_ejemplo.csv", index=False)

def create_demo_data():
    """Crear datos de demostración"""
    # Reutilizar la función existente pero mejorada
    np.random.seed(42)
    
    # Portfolio
    portfolio_data = []
    for i in range(30):
        quiniela = []
        for j in range(14):
            prob = np.random.random()
            if prob < 0.38:
                quiniela.append('L')
            elif prob < 0.68:
                quiniela.append('E') 
            else:
                quiniela.append('V')
        
        # Ajustar empates
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
    
    # Probabilities
    prob_data = []
    for i in range(14):
        p_l = np.random.uniform(0.25, 0.55)
        p_e = np.random.uniform(0.25, 0.35)
        p_v = 1.0 - p_l - p_e
        
        prob_data.append({
            'match_id': f'2283-{i+1}',
            'p_final_L': p_l,
            'p_final_E': p_e,
            'p_final_V': p_v
        })
    
    df_prob = pd.DataFrame(prob_data)
    df_prob.to_csv("data/dashboard/prob_draw_adjusted_2283.csv", index=False)
    
    # Simulation metrics
    sim_data = []
    for i in range(30):
        sim_data.append({
            'quiniela_id': f'Q{i+1}',
            'mu': np.random.uniform(8.5, 9.5),
            'sigma': np.random.uniform(1.0, 1.2),
            'pr_10': np.random.uniform(0.25, 0.35),
            'pr_11': np.random.uniform(0.08, 0.15)
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
    
    st.header(f"📊 Resultados - Jornada {jornada_id}")
    
    if not is_real_data:
        st.info("🎮 **Modo Demostración**: Los datos mostrados son sintéticos para fines ilustrativos")
    
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
    
    # Pestañas para diferentes vistas
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Portafolio", "📊 Análisis", "🎯 Probabilidades", "📈 Distribución"])
    
    with tab1:
        st.subheader("📋 Portafolio de 30 Quinielas")
        
        # Mostrar portafolio con colores
        df_port = data['portfolio']
        
        # Agregar métricas por quiniela
        df_display = df_port.merge(df_sim[['quiniela_id', 'pr_11', 'mu']], on='quiniela_id')
        
        # Formatear para display
        def color_quiniela(val):
            if val == 'L':
                return 'background-color: #e8f5e8'
            elif val == 'E':
                return 'background-color: #fff2cc'
            elif val == 'V':
                return 'background-color: #fce8e8'
            return ''
        
        # Aplicar estilos solo a columnas P1-P14
        cols_partidos = [f'P{i+1}' for i in range(14)]
        styled_df = df_display.style.applymap(color_quiniela, subset=cols_partidos)
        
        st.dataframe(styled_df, use_container_width=True, height=600)
        
        # Botón de descarga
        csv = df_port.to_csv(index=False)
        st.download_button(
            label="📥 Descargar Portafolio CSV",
            data=csv,
            file_name=f"portafolio_progol_{jornada_id}.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.subheader("📊 Análisis de Distribución")
        
        # Gráfico de distribución de signos
        signos = df_port.drop(columns='quiniela_id').values.flatten()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Distribución global
        unique, counts = np.unique(signos, return_counts=True)
        colors = ['#2e8b57', '#ffa500', '#dc143c']  # Verde, Naranja, Rojo
        bars = ax1.bar(unique, counts, color=colors[:len(unique)])
        ax1.set_xlabel('Signo')
        ax1.set_ylabel('Cantidad')
        ax1.set_title('Distribución Global de Signos')
        
        # Agregar porcentajes
        total = sum(counts)
        for i, (bar, count) in enumerate(zip(bars, counts)):
            pct = count/total*100
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    f'{count}\n({pct:.1f}%)', ha='center', va='bottom')
        
        # Distribución por quiniela
        empates_por_quiniela = []
        for _, row in df_port.iterrows():
            q = row.drop('quiniela_id').tolist()
            empates_por_quiniela.append(q.count('E'))
        
        ax2.hist(empates_por_quiniela, bins=range(3, 8), alpha=0.7, color='orange', edgecolor='black')
        ax2.axvline(4, color='red', linestyle='--', alpha=0.7, label='Mínimo (4)')
        ax2.axvline(6, color='red', linestyle='--', alpha=0.7, label='Máximo (6)')
        ax2.set_xlabel('Número de Empates')
        ax2.set_ylabel('Frecuencia')
        ax2.set_title('Distribución de Empates por Quiniela')
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Estadísticas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏠 Locales", f"{counts[unique == 'L'][0] if 'L' in unique else 0}")
        with col2:
            st.metric("🤝 Empates", f"{counts[unique == 'E'][0] if 'E' in unique else 0}")
        with col3:
            st.metric("✈️ Visitantes", f"{counts[unique == 'V'][0] if 'V' in unique else 0}")
    
    with tab3:
        st.subheader("🎯 Probabilidades por Partido")
        
        df_prob = data['probabilities']
        
        # Gráfico de probabilidades
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = range(1, 15)
        width = 0.25
        
        ax.bar([i - width for i in x], df_prob['p_final_L'], width, label='Local (L)', alpha=0.8, color='#2e8b57')
        ax.bar(x, df_prob['p_final_E'], width, label='Empate (E)', alpha=0.8, color='#ffa500')
        ax.bar([i + width for i in x], df_prob['p_final_V'], width, label='Visitante (V)', alpha=0.8, color='#dc143c')
        
        ax.set_xlabel('Partido')
        ax.set_ylabel('Probabilidad')
        ax.set_title('Probabilidades Finales por Partido')
        ax.set_xticks(x)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Tabla de probabilidades
        df_prob_display = df_prob.copy()
        df_prob_display['p_final_L'] = df_prob_display['p_final_L'].apply(lambda x: f"{x:.2%}")
        df_prob_display['p_final_E'] = df_prob_display['p_final_E'].apply(lambda x: f"{x:.2%}")
        df_prob_display['p_final_V'] = df_prob_display['p_final_V'].apply(lambda x: f"{x:.2%}")
        
        st.dataframe(df_prob_display, use_container_width=True)
    
    with tab4:
        st.subheader("📈 Distribución de Métricas")
        
        # Gráficos de simulación
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Pr[≥11]
        ax1.hist(df_sim['pr_11'], bins=15, alpha=0.7, color='green', edgecolor='black')
        ax1.axvline(df_sim['pr_11'].mean(), color='red', linestyle='--', 
                   label=f'Media: {df_sim["pr_11"].mean():.2%}')
        ax1.set_xlabel('Pr[≥11]')
        ax1.set_ylabel('Frecuencia')
        ax1.set_title('Distribución Pr[≥11] por Quiniela')
        ax1.legend()
        
        # μ hits
        ax2.hist(df_sim['mu'], bins=15, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(df_sim['mu'].mean(), color='red', linestyle='--',
                   label=f'Media: {df_sim["mu"].mean():.2f}')
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

if __name__ == "__main__":
    main()