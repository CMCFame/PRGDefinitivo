"""
Punto de entrada principal para Streamlit Cloud
Este archivo DEBE estar en la raíz del proyecto
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Configurar página
st.set_page_config(
    page_title="Progol Engine",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Agregar paths necesarios
sys.path.insert(0, os.path.dirname(__file__))

# Verificar si estamos en Streamlit Cloud
IN_STREAMLIT_CLOUD = os.getenv("STREAMLIT_CLOUD_ENVIRONMENT") is not None

# Si estamos en Streamlit Cloud, crear estructura mínima
if IN_STREAMLIT_CLOUD:
    # Crear directorios necesarios
    for dir_path in ["data/processed", "data/dashboard", "data/reports"]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Mostrar mensaje si no hay datos
    if not any(Path("data/dashboard").glob("*.csv")):
        st.warning("⚠️ Ejecutándose en Streamlit Cloud sin datos locales")
        st.info("""
        Para usar la aplicación completa necesitas:
        1. Ejecutar el pipeline localmente para generar datos
        2. Subir los archivos generados a `data/dashboard/`
        3. O conectar una base de datos PostgreSQL
        """)

# Importar y ejecutar el dashboard principal
try:
    from streamlit_app.dashboard import main
    main()
except FileNotFoundError as e:
    st.error(f"Error: No se encontraron archivos de datos necesarios")
    st.info("Por favor ejecuta el pipeline primero: `make run-all`")
    st.code("""
    # Instalación local:
    git clone <repo>
    cd progol-engine
    python setup.py
    make run-all
    streamlit run streamlit_app.py
    """)
except Exception as e:
    st.error(f"Error inesperado: {str(e)}")
    st.info("Verifica que todas las dependencias estén instaladas")