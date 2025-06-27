#!/usr/bin/env python3
"""
Script de verificación de instalación de Progol Engine
Ejecutar: python scripts/verify_installation.py
"""

import os
import sys
import importlib
from pathlib import Path

# Colores para output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
END = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*60}{END}")
    print(f"{BLUE}  {text}{END}")
    print(f"{BLUE}{'='*60}{END}")

def check_mark(condition, message):
    if condition:
        print(f"{GREEN}✓{END} {message}")
        return True
    else:
        print(f"{RED}✗{END} {message}")
        return False

def check_python_version():
    """Verifica versión de Python"""
    version = sys.version_info
    return check_mark(
        version.major >= 3 and version.minor >= 8,
        f"Python {version.major}.{version.minor} (se requiere 3.8+)"
    )

def check_directory_structure():
    """Verifica estructura de directorios"""
    print_header("Verificando Estructura de Directorios")
    
    required_dirs = [
        "data/raw",
        "data/processed",
        "data/dashboard",
        "data/reports",
        "data/json_previas",
        "models/poisson",
        "models/bayes",
        "src/etl",
        "src/modeling",
        "src/optimizer",
        "src/utils",
        "streamlit_app",
        "tests",
        "airflow/dags",
        "notebooks"
    ]
    
    all_good = True
    for dir_path in required_dirs:
        exists = Path(dir_path).exists()
        check_mark(exists, f"Directorio: {dir_path}")
        all_good = all_good and exists
    
    return all_good

def check_required_files():
    """Verifica archivos requeridos"""
    print_header("Verificando Archivos Principales")
    
    required_files = [
        ("requirements.txt", "Dependencias"),
        (".env.template", "Template de configuración"),
        ("setup.py", "Script de setup"),
        ("streamlit_app/dashboard.py", "Dashboard principal"),
        ("src/utils/config.py", "Configuración"),
        ("src/utils/logger.py", "Logger"),
        ("src/etl/ingest_csv.py", "ETL: Ingesta"),
        ("src/modeling/poisson_model.py", "Modelo Poisson"),
        ("src/optimizer/grasp.py", "Optimizador GRASP"),
        ("streamlit_app.py", "Punto de entrada Streamlit Cloud")
    ]
    
    all_good = True
    for file_path, description in required_files:
        exists = Path(file_path).exists()
        check_mark(exists, f"{description}: {file_path}")
        all_good = all_good and exists
        
        # Advertencia especial para streamlit_app.py
        if file_path == "streamlit_app.py" and not exists:
            print(f"{YELLOW}  ⚠️  Este archivo es NECESARIO para Streamlit Cloud{END}")
    
    return all_good

def check_dependencies():
    """Verifica dependencias instaladas"""
    print_header("Verificando Dependencias Python")
    
    critical_packages = [
        ("pandas", "Manipulación de datos"),
        ("numpy", "Cálculos numéricos"),
        ("streamlit", "Dashboard"),
        ("matplotlib", "Visualizaciones"),
        ("scipy", "Estadísticas"),
        ("sklearn", "Machine Learning"),
    ]
    
    optional_packages = [
        ("airflow", "Orquestación"),
        ("pytest", "Testing"),
        ("cmdstanpy", "Modelos bayesianos"),
    ]
    
    all_critical = True
    
    print("\n📦 Paquetes Críticos:")
    for package, description in critical_packages:
        try:
            importlib.import_module(package)
            check_mark(True, f"{package} - {description}")
        except ImportError:
            check_mark(False, f"{package} - {description}")
            all_critical = False
    
    print("\n📦 Paquetes Opcionales:")
    for package, description in optional_packages:
        try:
            importlib.import_module(package)
            check_mark(True, f"{package} - {description}")
        except ImportError:
            print(f"{YELLOW}⚠{END}  {package} - {description} (opcional)")
    
    return all_critical

def check_environment():
    """Verifica configuración de entorno"""
    print_header("Verificando Configuración")
    
    env_exists = Path(".env").exists()
    check_mark(env_exists, "Archivo .env existe")
    
    if not env_exists and Path(".env.template").exists():
        print(f"{YELLOW}  → Ejecuta: cp .env.template .env{END}")
    
    # Verificar si estamos en un entorno virtual
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    check_mark(in_venv, "Entorno virtual activado")
    
    if not in_venv:
        print(f"{YELLOW}  → Recomendado: python -m venv venv && source venv/bin/activate{END}")
    
    return env_exists

def check_sample_data():
    """Verifica si hay datos de ejemplo"""
    print_header("Verificando Datos")
    
    sample_files = [
        "data/raw/Progol_sample.csv",
        "data/raw/odds_sample.csv",
        "data/json_previas/previas_sample.json"
    ]
    
    any_data = False
    for file_path in sample_files:
        exists = Path(file_path).exists()
        if exists:
            check_mark(True, f"Datos de ejemplo: {file_path}")
            any_data = True
    
    if not any_data:
        print(f"{YELLOW}⚠  No se encontraron datos de ejemplo{END}")
        print(f"{YELLOW}  → Ejecuta: python setup.py (opción crear datos de ejemplo){END}")
    
    return any_data

def test_imports():
    """Prueba importar módulos principales"""
    print_header("Probando Imports")
    
    test_modules = [
        "src.utils.config",
        "src.utils.logger",
        "src.etl.ingest_csv",
        "src.modeling.poisson_model",
        "src.optimizer.grasp",
    ]
    
    # Agregar src al path
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    all_good = True
    for module in test_modules:
        try:
            importlib.import_module(module)
            check_mark(True, f"Import: {module}")
        except Exception as e:
            check_mark(False, f"Import: {module} - {str(e)}")
            all_good = False
    
    return all_good

def main():
    """Función principal de verificación"""
    print(f"\n{BLUE}🔍 VERIFICACIÓN DE INSTALACIÓN - PROGOL ENGINE{END}")
    
    results = {
        "Python": check_python_version(),
        "Directorios": check_directory_structure(),
        "Archivos": check_required_files(),
        "Dependencias": check_dependencies(),
        "Entorno": check_environment(),
        "Datos": check_sample_data(),
        "Imports": test_imports()
    }
    
    # Resumen
    print_header("RESUMEN")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nPruebas pasadas: {passed}/{total}")
    
    if passed == total:
        print(f"\n{GREEN}✅ ¡TODO ESTÁ LISTO!{END}")
        print(f"\nPuedes ejecutar:")
        print(f"  {BLUE}streamlit run streamlit_app/dashboard.py{END}")
        print(f"  o")
        print(f"  {BLUE}make run-all{END}")
    else:
        print(f"\n{YELLOW}⚠️  Hay algunos problemas que resolver{END}")
        print(f"\nProblemas detectados:")
        for check, passed in results.items():
            if not passed:
                print(f"  - {check}")
        
        print(f"\n{YELLOW}Sugerencias:{END}")
        if not results["Dependencias"]:
            print(f"  1. Instala dependencias: {BLUE}pip install -r requirements.txt{END}")
        if not results["Archivos"]:
            print(f"  2. Verifica que todos los archivos estén en su lugar")
        if not results["Directorios"]:
            print(f"  3. Ejecuta: {BLUE}python setup.py{END}")
    
    print()

if __name__ == "__main__":
    main()