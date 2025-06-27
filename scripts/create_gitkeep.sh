#!/bin/bash

# Script para crear archivos .gitkeep en directorios vacíos
# Esto permite mantener la estructura de carpetas en Git

echo "Creando archivos .gitkeep..."

# Lista de directorios que necesitan .gitkeep
directories=(
    "data/raw"
    "data/processed"
    "data/dashboard"
    "data/reports"
    "data/analysis"
    "data/json_previas"
    "models/poisson"
    "models/bayes"
    "logs"
    "scripts"
    "monitoring/prometheus"
    "monitoring/grafana/dashboards"
    "monitoring/grafana/datasources"
)

# Crear directorios y archivos .gitkeep
for dir in "${directories[@]}"; do
    mkdir -p "$dir"
    touch "$dir/.gitkeep"
    echo "✓ Creado: $dir/.gitkeep"
done

echo ""
echo "✅ Archivos .gitkeep creados exitosamente!"
echo ""
echo "Nota: Los archivos .gitkeep permiten que Git trackee carpetas vacías."