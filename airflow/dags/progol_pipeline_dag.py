from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os

default_args = {
    'owner': 'progol',
    'start_date': datetime(2025, 1, 1),
    'retries': 0,
}

dag = DAG(
    'progol_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
    description='Pipeline de generación y validación de portafolio Progol',
)

def run_script(path):
    os.system(f'python {path}')

etapas = [
    ("parse_previas",        "src/etl/parse_previas.py"),
    ("build_features",       "src/etl/build_features.py"),
    ("poisson_model",        "src/modeling/poisson_model.py"),
    ("stacking",             "src/modeling/stacking.py"),
    ("bayesian_adjustment",  "src/modeling/bayesian_adjustment.py"),
    ("draw_propensity",      "src/modeling/draw_propensity.py"),
    ("classify_matches",     "src/optimizer/classify_matches.py"),
    ("generate_core",        "src/optimizer/generate_core.py"),
    ("generate_satellites",  "src/optimizer/generate_satellites.py"),
    ("grasp",                "src/optimizer/grasp.py"),
    ("annealing",            "src/optimizer/annealing.py"),
    ("checklist",            "src/optimizer/checklist.py"),
    ("montecarlo_sim",       "src/simulation/montecarlo_sim.py"),
]

prev_task = None
for name, script in etapas:
    task = PythonOperator(
        task_id=name,
        python_callable=run_script,
        op_args=[script],
        dag=dag,
    )
    if prev_task:
        prev_task >> task
    prev_task = task
