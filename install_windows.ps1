# ============================================================================
# PROGOL ENGINE - SCRIPT DE INSTALACIÓN PARA WINDOWS
# ============================================================================

Write-Host "🪟 PROGOL ENGINE - INSTALACIÓN WINDOWS" -ForegroundColor Blue
Write-Host "=======================================" -ForegroundColor Blue
Write-Host ""

# Verificar Python
Write-Host "🔍 Verificando Python..." -ForegroundColor Yellow
$pythonVersion = python --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Python no encontrado. Instala Python 3.11+ desde:" -ForegroundColor Red
    Write-Host "   https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}
Write-Host "✅ $pythonVersion encontrado" -ForegroundColor Green

# Verificar si estamos en entorno virtual
if ($env:VIRTUAL_ENV) {
    Write-Host "✅ Entorno virtual activo: $env:VIRTUAL_ENV" -ForegroundColor Green
} else {
    Write-Host "⚠️  No estás en un entorno virtual. Creando uno..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "🔄 Activando entorno virtual..." -ForegroundColor Yellow
    .\venv\Scripts\Activate.ps1
}

# Actualizar pip
Write-Host "🔄 Actualizando pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

# Instalar Microsoft Visual C++ Build Tools (necesario para algunas dependencias)
Write-Host "🔍 Verificando Visual C++ Build Tools..." -ForegroundColor Yellow
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vsWhere) {
    Write-Host "✅ Visual Studio Build Tools encontrado" -ForegroundColor Green
} else {
    Write-Host "⚠️  Visual C++ Build Tools no encontrado" -ForegroundColor Yellow
    Write-Host "   Algunas dependencias pueden fallar al compilar" -ForegroundColor Yellow
    Write-Host "   Descarga desde: https://visualstudio.microsoft.com/visual-cpp-build-tools/" -ForegroundColor Yellow
    $continue = Read-Host "¿Continuar de todos modos? (y/n)"
    if ($continue -ne "y") { exit 1 }
}

# Instalar dependencias por grupos para mejor manejo de errores
Write-Host "📦 Instalando dependencias básicas..." -ForegroundColor Yellow
$basicPackages = @(
    "pandas==2.1.4",
    "numpy==1.24.3", 
    "scipy==1.11.1",
    "matplotlib==3.7.2",
    "seaborn==0.12.2",
    "plotly==5.15.0",
    "streamlit==1.28.1"
)

foreach ($package in $basicPackages) {
    Write-Host "  Instalando $package..." -ForegroundColor Cyan
    pip install $package
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Error instalando $package" -ForegroundColor Red
        exit 1
    }
}

Write-Host "📦 Instalando dependencias de ciencia de datos..." -ForegroundColor Yellow
$dataPackages = @(
    "scikit-learn==1.3.0",
    "statsmodels==0.14.0",
    "joblib==1.3.1",
    "pyarrow==13.0.0"
)

foreach ($package in $dataPackages) {
    Write-Host "  Instalando $package..." -ForegroundColor Cyan
    pip install $package
}

Write-Host "📦 Instalando dependencias de base de datos..." -ForegroundColor Yellow
pip install psycopg2-binary==2.9.7 sqlalchemy==2.0.23

Write-Host "📦 Instalando utilidades..." -ForegroundColor Yellow
$utilPackages = @(
    "python-dotenv==1.0.0",
    "click==8.1.7",
    "tqdm==4.66.1",
    "requests==2.31.0",
    "beautifulsoup4==4.12.2"
)

foreach ($package in $utilPackages) {
    pip install $package
}

# Instalar Airflow (puede ser problemático en Windows)
Write-Host "📦 Instalando Apache Airflow..." -ForegroundColor Yellow
Write-Host "   (Esto puede tomar varios minutos en Windows)" -ForegroundColor Cyan

# Configurar constraints para Airflow en Windows
$constraintUrl = "https://raw.githubusercontent.com/apache/airflow/constraints-2.7.3/constraints-3.11.txt"
pip install "apache-airflow==2.7.3" --constraint $constraintUrl

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Airflow falló. Instalando sin constraints..." -ForegroundColor Yellow
    pip install apache-airflow==2.7.3 --no-deps
    pip install apache-airflow-providers-postgres==5.6.0
}

# Instalar dependencias opcionales
Write-Host "📦 Instalando dependencias avanzadas..." -ForegroundColor Yellow
$advancedPackages = @(
    "pytest==7.4.3",
    "black==23.11.0",
    "flake8==6.1.0",
    "jupyter==1.0.0",
    "notebook==7.0.6"
)

foreach ($package in $advancedPackages) {
    pip install $package --no-warn-script-location
}

# Instalar Stan (puede fallar en Windows)
Write-Host "📦 Instalando Stan (Modelos Bayesianos)..." -ForegroundColor Yellow
Write-Host "   (Esto puede fallar en algunas configuraciones de Windows)" -ForegroundColor Cyan

pip install cmdstanpy==1.2.0
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  cmdstanpy falló. Intentando pystan..." -ForegroundColor Yellow
    pip install pystan==3.7.0
    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠️  Stan no se pudo instalar. Continuando sin modelos Bayesianos..." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "✅ INSTALACIÓN DE DEPENDENCIAS COMPLETA!" -ForegroundColor Green
Write-Host ""

# Verificar instalación
Write-Host "🔍 Verificando instalación..." -ForegroundColor Yellow
$testImports = @(
    "import pandas; print('✅ pandas:', pandas.__version__)",
    "import numpy; print('✅ numpy:', numpy.__version__)",
    "import streamlit; print('✅ streamlit:', streamlit.__version__)",
    "import sklearn; print('✅ scikit-learn:', sklearn.__version__)"
)

foreach ($test in $testImports) {
    python -c $test
}

Write-Host ""
Write-Host "🎯 SIGUIENTE PASO: Configurar el proyecto" -ForegroundColor Blue
Write-Host "   python setup.py" -ForegroundColor Cyan
Write-Host ""