# 🔢 Progol Engine – Plataforma Algorítmica para Quinielas

Este repositorio implementa la versión operativa de la **Metodología Definitiva Progol**, un framework cuantitativo y heurístico para maximizar la probabilidad de ≥11 aciertos en quinielas tipo Progol.

## 🎯 Objetivo

Construir un sistema reproducible, optimizado y basado en evidencia que:
- Integre datos históricos, momios y previas.
- Modele probabilidades calibradas por partido.
- Genere 30 quinielas diversificadas (4 Core + 26 Satélites).
- Optimice la distribución mediante GRASP + Annealing.
- Evalúe y visualice métricas clave (Pr[≥11], ROI, Varianza).

## 🧱 Estructura del Repositorio

- `data/`: Fuentes crudas y procesadas (Progol, odds, previas).
- `models/`: Pesos y parámetros entrenados (Poisson, Bayes).
- `src/`: Código fuente organizado por bloque (ETL, modelado, optimización).
- `airflow/`: DAGs automatizados para la operación diaria.
- `streamlit_app/`: Visualización e interfaz operativa.
- `tests/`: Validación unitaria del sistema completo.
- `notebooks/`: Exploración y experimentación.

## ⚙️ Pipeline

1. **ETL**: ingestión y limpieza de datos.
2. **Modelado**: Poisson Bivariado + Calibración Bayesiana.
3. **Generación**: arquitectura Core + Satélites.
4. **Optimización**: GRASP + Simulated Annealing.
5. **Exportación**: portafolio final + métricas.
6. **Visualización**: dashboard e indicadores en tiempo real.

## 🛠️ Instalación

```bash
git clone https://github.com/<usuario>/progol-engine.git
cd progol-engine
python -m venv venv
source venv/bin/activate  # en Linux/Mac
pip install -r requirements.txt
