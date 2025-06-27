#!/usr/bin/env python
"""
Script de configuración inicial para Progol Engine
Ejecutar después de clonar el repositorio
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import json
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def print_header(text):
    """Imprime encabezado formateado"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def check_python_version():
    """Verifica versión de Python"""
    print_header("Verificando Python")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print(f"❌ Python {version.major}.{version.minor} detectado")
        print("   Se requiere Python 3.11 o superior")
        return False
    
    print(f"✅ Python {version.major}.{version.minor} detectado")
    return True


def create_directories():
    """Crea estructura de directorios"""
    print_header("Creando directorios")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/dashboard",
        "data/reports",
        "data/analysis",
        "data/json_previas",
        "models/poisson",
        "models/bayes",
        "logs",
        ".streamlit"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Creado: {dir_path}")


def setup_env_file():
    """Configura archivo .env"""
    print_header("Configurando variables de entorno")
    
    if Path(".env").exists():
        print("⚠️  .env ya existe, saltando...")
        return
    
    if Path(".env.template").exists():
        shutil.copy(".env.template", ".env")
        print("✅ .env creado desde template")
        print("   Por favor edita .env con tu configuración")
    else:
        # Crear .env básico
        env_content = """# === ENTORNO DE DATOS ===
DATA_RAW_PATH=data/raw/
DATA_PROCESSED_PATH=data/processed/
JSON_PREVIAS_PATH=data/json_previas/

# === PARÁMETROS DEL PIPELINE ===
N_QUINIELAS=30
COSTO_BOLETO=15
PREMIO_CAT2=90000
N_MONTECARLO_SAMPLES=50000
PR11_UMBRAL=0.10

# === COEFICIENTES BAYESIANOS DEFAULT ===
K1_L=0.15
K2_L=-0.12
K3_L=0.08
K1_E=-0.10
K2_E=0.15
K3_E=0.03
K1_V=-0.08
K2_V=-0.10
K3_V=-0.05

# === BASE DE DATOS (opcional) ===
PG_HOST=localhost
PG_PORT=5432
PG_DB=progol
PG_USER=progol_admin
PG_PASSWORD=changeme

# === STREAMLIT DASHBOARD ===
STREAMLIT_PORT=8501
DASHBOARD_REFRESH_INTERVAL=5

# === LOGS & ALERTAS ===
ENABLE_ALERTS=false
SLACK_WEBHOOK_URL=
EMAIL_NOTIFIER=

# === ARCHIVOS DEFAULT POR JORNADA ===
JORNADA_ID=2283
PREVIAS_PDF=previas_2283.pdf
PROGOL_CSV=Progol.csv
ODDS_CSV=odds.csv
ELO_CSV=elo.csv
SQUAD_CSV=squad_value.csv
"""
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ .env creado con valores default")


def create_sample_data():
    """Crea datos de ejemplo para testing"""
    print_header("Creando datos de ejemplo")
    
    jornada_id = 2283
    fecha = datetime.now().strftime("%Y-%m-%d")
    
    # Equipos de ejemplo
    equipos_liga_mx = [
        "América", "Chivas", "Cruz Azul", "Pumas", "Tigres", "Monterrey",
        "Santos", "Toluca", "León", "Pachuca", "Atlas", "Necaxa",
        "Puebla", "Querétaro", "Tijuana", "Juárez", "Mazatlán", "San Luis"
    ]
    
    # Crear Progol.csv de ejemplo
    progol_data = []
    for i in range(1, 15):
        home_idx = (i-1) * 2 % len(equipos_liga_mx)
        away_idx = (home_idx + 1) % len(equipos_liga_mx)
        
        goles_h = random.randint(0, 3)
        goles_a = random.randint(0, 3)
        
        if goles_h > goles_a:
            resultado = 'L'
        elif goles_h < goles_a:
            resultado = 'V'
        else:
            resultado = 'E'
        
        progol_data.append({
            'concurso_id': jornada_id,
            'fecha': fecha,
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
    
    df_progol = pd.DataFrame(progol_data)
    df_progol.to_csv("data/raw/Progol_sample.csv", index=False)
    print("✅ Creado: data/raw/Progol_sample.csv")
    
    # Crear odds.csv de ejemplo
    odds_data = []
    for i, partido in enumerate(progol_data, 1):
        # Generar momios pseudo-realistas
        if partido['resultado'] == 'L':
            odds_l = round(random.uniform(1.5, 2.5), 2)
            odds_e = round(random.uniform(3.0, 4.0), 2)
            odds_v = round(random.uniform(3.5, 5.0), 2)
        elif partido['resultado'] == 'V':
            odds_l = round(random.uniform(3.5, 5.0), 2)
            odds_e = round(random.uniform(3.0, 4.0), 2)
            odds_v = round(random.uniform(1.5, 2.5), 2)
        else:
            odds_l = round(random.uniform(2.5, 3.5), 2)
            odds_e = round(random.uniform(2.8, 3.5), 2)
            odds_v = round(random.uniform(2.5, 3.5), 2)
        
        odds_data.append({
            'concurso_id': jornada_id,
            'match_no': i,
            'fecha': fecha,
            'home': partido['home'],
            'away': partido['away'],
            'odds_L': odds_l,
            'odds_E': odds_e,
            'odds_V': odds_v
        })
    
    df_odds = pd.DataFrame(odds_data)
    df_odds.to_csv("data/raw/odds_sample.csv", index=False)
    print("✅ Creado: data/raw/odds_sample.csv")
    
    # Crear previas JSON de ejemplo
    previas = []
    for i in range(1, 15):
        forma_h = ''.join(random.choices(['W', 'D', 'L'], k=5))
        forma_a = ''.join(random.choices(['W', 'D', 'L'], k=5))
        
        h2h_total = 5
        h2h_h = random.randint(0, 3)
        h2h_a = random.randint(0, 3)
        h2h_e = h2h_total - h2h_h - h2h_a
        
        previas.append({
            "match_id": f"{jornada_id}-{i}",
            "form_H": forma_h,
            "form_A": forma_a,
            "h2h_H": h2h_h,
            "h2h_E": max(0, h2h_e),
            "h2h_A": h2h_a,
            "inj_H": random.randint(0, 2),
            "inj_A": random.randint(0, 2),
            "context_flag": ["derbi"] if i in [1, 7] else []
        })
    
    with open("data/json_previas/previas_sample.json", "w") as f:
        json.dump(previas, f, indent=2)
    print("✅ Creado: data/json_previas/previas_sample.json")
    
    # Crear datos Elo de ejemplo
    elo_data = []
    for equipo in equipos_liga_mx[:14]:
        elo_data.append({
            'team': equipo,
            'elo': random.randint(1500, 1700),
            'fecha': fecha
        })
    
    df_elo = pd.DataFrame(elo_data)
    
    # Expandir para home/away
    elo_matches = []
    for partido in progol_data:
        elo_h = df_elo[df_elo['team'] == partido['home']]['elo'].values[0] if partido['home'] in df_elo['team'].values else 1600
        elo_a = df_elo[df_elo['team'] == partido['away']]['elo'].values[0] if partido['away'] in df_elo['team'].values else 1600
        
        elo_matches.append({
            'home': partido['home'],
            'away': partido['away'],
            'fecha': fecha,
            'elo_home': elo_h,
            'elo_away': elo_a,
            'factor_local': 0.45
        })
    
    df_elo_matches = pd.DataFrame(elo_matches)
    df_elo_matches.to_csv("data/raw/elo_sample.csv", index=False)
    print("✅ Creado: data/raw/elo_sample.csv")


def setup_airflow():
    """Configura Airflow básico"""
    print_header("Configurando Airflow")
    
    try:
        # Verificar si Airflow está instalado
        import airflow
        print("✅ Airflow detectado")
        
        # Crear directorio de DAGs si no existe
        airflow_home = os.environ.get('AIRFLOW_HOME', os.path.expanduser('~/airflow'))
        dags_folder = Path(airflow_home) / 'dags'
        dags_folder.mkdir(parents=True, exist_ok=True)
        
        # Crear link simbólico a nuestros DAGs
        our_dags = Path('airflow/dags').absolute()
        for dag_file in our_dags.glob('*.py'):
            link_path = dags_folder / dag_file.name
            if not link_path.exists():
                try:
                    link_path.symlink_to(dag_file)
                    print(f"✅ Link creado: {dag_file.name}")
                except:
                    shutil.copy(dag_file, link_path)
                    print(f"✅ Copiado: {dag_file.name}")
        
        print("\n💡 Para inicializar Airflow:")
        print("   airflow db init")
        print("   airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com")
        print("   airflow webserver --port 8080 &")
        print("   airflow scheduler &")
        
    except ImportError:
        print("⚠️  Airflow no instalado")
        print("   Los DAGs están en airflow/dags/ para cuando lo instales")


def setup_streamlit():
    """Configura Streamlit"""
    print_header("Configurando Streamlit")
    
    # Crear config de Streamlit
    streamlit_config = """[theme]
primaryColor = "#2ecc71"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
port = 8501

[browser]
gatherUsageStats = false
"""
    
    config_path = Path(".streamlit/config.toml")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, "w") as f:
        f.write(streamlit_config)
    
    print("✅ Configuración de Streamlit creada")
    print("\n💡 Para iniciar el dashboard:")
    print("   streamlit run streamlit_app/dashboard.py")


def install_dependencies():
    """Instala dependencias opcionales"""
    print_header("Verificando dependencias")
    
    print("📦 Las dependencias principales se instalan con:")
    print("   pip install -r requirements.txt")
    
    # Verificar algunas dependencias clave
    try:
        import pandas
        print("✅ pandas instalado")
    except:
        print("❌ pandas no instalado")
    
    try:
        import numpy
        print("✅ numpy instalado")
    except:
        print("❌ numpy no instalado")
    
    try:
        import streamlit
        print("✅ streamlit instalado")
    except:
        print("❌ streamlit no instalado")
    
    try:
        import airflow
        print("✅ airflow instalado")
    except:
        print("⚠️  airflow no instalado (opcional)")


def run_tests():
    """Ejecuta tests básicos"""
    print_header("Ejecutando tests de verificación")
    
    try:
        subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"], 
                      capture_output=True, text=True, check=True)
        print("✅ Todos los tests pasaron")
    except subprocess.CalledProcessError as e:
        print("⚠️  Algunos tests fallaron (normal en setup inicial)")
        print("   Ejecuta 'pytest tests/ -v' para más detalles")
    except FileNotFoundError:
        print("⚠️  pytest no instalado")
        print("   Instala con: pip install pytest")


def print_next_steps():
    """Imprime siguientes pasos"""
    print_header("✨ Setup Completado! ✨")
    
    print("\n📋 Siguientes pasos:\n")
    
    print("1. Editar configuración:")
    print("   - Edita .env con tu configuración")
    print("   - Revisa src/utils/config.py")
    
    print("\n2. Preparar datos reales:")
    print("   - Copia tu Progol.csv a data/raw/")
    print("   - Copia odds.csv a data/raw/")
    print("   - Copia PDFs de previas a data/raw/")
    
    print("\n3. Ejecutar pipeline manualmente:")
    print("   python -m src.etl.ingest_csv")
    print("   python -m src.modeling.poisson_model")
    print("   python -m src.optimizer.generate_core")
    
    print("\n4. O usar Airflow (recomendado):")
    print("   - Inicializar: airflow db init")
    print("   - Crear usuario: airflow users create ...")
    print("   - Iniciar: airflow webserver & airflow scheduler")
    
    print("\n5. Ver resultados:")
    print("   streamlit run streamlit_app/dashboard.py")
    
    print("\n📚 Documentación completa en README.md")
    print("\n¡Buena suerte con tus quinielas! 🍀")


def main():
    """Función principal"""
    print("\n🔢 PROGOL ENGINE - SETUP INICIAL 🔢")
    print("Version 1.0 - Metodología Definitiva")
    
    # Verificaciones
    if not check_python_version():
        print("\n❌ Setup cancelado. Instala Python 3.11+")
        sys.exit(1)
    
    # Setup
    create_directories()
    setup_env_file()
    create_sample_data()
    setup_airflow()
    setup_streamlit()
    install_dependencies()
    
    # Tests opcionales
    response = input("\n¿Ejecutar tests de verificación? (s/N): ")
    if response.lower() == 's':
        run_tests()
    
    # Finalizar
    print_next_steps()


if __name__ == "__main__":
    main()