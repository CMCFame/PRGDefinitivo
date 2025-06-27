"""
Punto de entrada principal para Streamlit Cloud
Este archivo DEBE estar en la raíz del proyecto
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Configurar página PRIMERO
st.set_page_config(
    page_title="Progol Engine",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Agregar paths
sys.path.insert(0, os.path.dirname(__file__))

# Verificar si estamos en Streamlit Cloud
IN_STREAMLIT_CLOUD = os.getenv("STREAMLIT_CLOUD_ENVIRONMENT") is not None

def create_sample_data():
    """Crear datos de ejemplo para demostración"""
    import pandas as pd
    import numpy as np
    from datetime import datetime
    
    # Crear directorio si no existe
    Path("data/dashboard").mkdir(parents=True, exist_ok=True)
    
    # Datos de ejemplo del portafolio
    np.random.seed(42)
    portfolio_data = []
    
    for i in range(30):
        quiniela = []
        # Generar una quiniela con distribución realista
        for j in range(14):
            prob = np.random.random()
            if prob < 0.38:
                quiniela.append('L')
            elif prob < 0.38 + 0.30:
                quiniela.append('E') 
            else:
                quiniela.append('V')
        
        # Asegurar 4-6 empates
        empates = quiniela.count('E')
        if empates < 4:
            for k in range(4 - empates):
                if k < len(quiniela):
                    quiniela[k] = 'E'
        elif empates > 6:
            count = 0
            for k in range(len(quiniela)):
                if quiniela[k] == 'E' and count < empates - 6:
                    quiniela[k] = 'L'
                    count += 1
        
        portfolio_data.append([f"Q{i+1}"] + quiniela)
    
    # Crear DataFrame
    cols = ['quiniela_id'] + [f'P{i+1}' for i in range(14)]
    df_portfolio = pd.DataFrame(portfolio_data, columns=cols)
    df_portfolio.to_csv("data/dashboard/portfolio_final_2283.csv", index=False)
    
    # Probabilidades de ejemplo
    prob_data = []
    for i in range(14):
        # Probabilidades realistas
        p_l = np.random.uniform(0.2, 0.6)
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
    
    # Métricas de simulación de ejemplo
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
    
    return True

def main():
    """Función principal"""
    st.title("🔢 Progol Engine Dashboard")
    
    # Si no hay datos, crear datos de ejemplo
    if not Path("data/dashboard/portfolio_final_2283.csv").exists():
        with st.spinner("Creando datos de ejemplo..."):
            create_sample_data()
        st.success("✅ Datos de ejemplo creados")
        st.rerun()
    
    try:
        # Intentar cargar el dashboard real
        from streamlit_app.dashboard import main as dashboard_main
        dashboard_main()
        
    except (ImportError, FileNotFoundError) as e:
        # Fallback: Dashboard simple con datos de ejemplo
        st.warning("⚠️ Ejecutándose en modo demostración con datos de ejemplo")
        
        # Cargar datos de ejemplo
        try:
            import pandas as pd
            
            df_portfolio = pd.read_csv("data/dashboard/portfolio_final_2283.csv")
            df_prob = pd.read_csv("data/dashboard/prob_draw_adjusted_2283.csv") 
            df_sim = pd.read_csv("data/dashboard/simulation_metrics_2283.csv")
            
            # Mostrar información básica
            st.subheader("📋 Portafolio de 30 Quinielas")
            st.dataframe(df_portfolio, use_container_width=True)
            
            # Métricas principales
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Quinielas", len(df_portfolio))
            
            with col2:
                pr11 = df_sim['pr_11'].mean()
                st.metric("🎯 Pr[≥11]", f"{pr11:.2%}")
            
            with col3:
                mu_hits = df_sim['mu'].mean()
                st.metric("🔢 μ hits", f"{mu_hits:.2f}")
            
            with col4:
                sigma_hits = df_sim['sigma'].mean()
                st.metric("📈 σ hits", f"{sigma_hits:.2f}")
            
            # Gráfico de distribución
            st.subheader("📊 Distribución de Signos")
            
            # Contar signos
            signos = df_portfolio.drop(columns='quiniela_id').values.flatten()
            import matplotlib.pyplot as plt
            import numpy as np  # ← AGREGAR ESTA LÍNEA
            
            fig, ax = plt.subplots(figsize=(8, 5))
            unique, counts = np.unique(signos, return_counts=True)
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            ax.bar(unique, counts, color=colors[:len(unique)])
            ax.set_xlabel('Signo')
            ax.set_ylabel('Cantidad')
            ax.set_title('Distribución de Signos en las 30 Quinielas')
            
            # Agregar porcentajes
            total = sum(counts)
            for i, (label, count) in enumerate(zip(unique, counts)):
                pct = count/total*100
                ax.text(i, count + 5, f'{count}\n({pct:.1f}%)', 
                       ha='center', va='bottom')
            
            st.pyplot(fig)
            
            st.info("""
            **📝 Nota:** Esta es una demostración con datos sintéticos.
            
            Para usar con datos reales:
            1. Ejecuta el pipeline completo localmente
            2. Sube los archivos CSV generados a `data/dashboard/`
            3. O conecta una base de datos PostgreSQL
            """)
            
        except Exception as inner_e:
            st.error(f"Error cargando datos de ejemplo: {inner_e}")
            st.code("""
            # Para ejecutar localmente:
            git clone <tu-repo>
            cd progol-engine
            python setup.py
            streamlit run streamlit_app.py
            """)

if __name__ == "__main__":
    main()