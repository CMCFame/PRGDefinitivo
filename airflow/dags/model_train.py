from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.task_group import TaskGroup
from airflow.sensors.filesystem import FileSensor
from datetime import datetime, timedelta
import os
import sys
import numpy as np
import pandas as pd
import json

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src.modeling.poisson_model import (preparar_dataset_poisson, entrenar_poisson_model,
                                       predecir_lambdas, guardar_lambda)
from src.modeling.stacking import stack_probabilities, guardar_blend
from src.modeling.bayesian_adjustment import ajustar_bayes, guardar_final
from src.modeling.draw_propensity import aplicar_draw_propensity, guardar_prob_draw
from src.utils.logger import get_logger
from src.utils.config import JORNADA_ID, N_MONTECARLO
from src.utils.validators import validar_probabilidades

logger = get_logger("model_train_dag")

default_args = {
    'owner': 'progol',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=10)
}

dag = DAG(
    'model_training_pipeline',
    default_args=default_args,
    description='Pipeline de entrenamiento de modelos Progol',
    schedule_interval='0 1 * * *',  # 1 AM diario
    catchup=False,
    tags=['ml', 'modeling', 'progol']
)

# Sensor para esperar que ETL termine
wait_for_features = FileSensor(
    task_id='wait_for_features',
    filepath=f"data/processed/match_features_{JORNADA_ID}.feather",
    poke_interval=30,
    timeout=300,
    dag=dag
)

start = DummyOperator(task_id='start', dag=dag)

# Grupo de modelado Poisson
with TaskGroup('poisson_modeling', dag=dag) as poisson_group:
    
    def _train_poisson():
        """Entrena modelo Poisson bivariado"""
        logger.info("Iniciando entrenamiento Poisson")
        
        # Cargar datos históricos + jornada actual
        df_features = pd.read_feather(f"data/processed/match_features_{JORNADA_ID}.feather")
        
        # Para entrenamiento, necesitamos datos históricos
        # En producción, esto vendría de una BD
        try:
            df_historical = pd.read_csv("data/raw/progol_historical.csv")
            df_train = pd.concat([df_historical, df_features], ignore_index=True)
        except:
            logger.warning("Sin datos históricos, usando solo jornada actual")
            df_train = df_features
        
        # Preparar dataset
        df_train = preparar_dataset_poisson(df_train)
        
        # Entrenar modelos
        model_H, model_A, ohe = entrenar_poisson_model(df_train)
        
        # Guardar modelos
        import joblib
        os.makedirs("models/poisson", exist_ok=True)
        joblib.dump(model_H, f"models/poisson/model_H_{JORNADA_ID}.pkl")
        joblib.dump(model_A, f"models/poisson/model_A_{JORNADA_ID}.pkl")
        joblib.dump(ohe, f"models/poisson/ohe_{JORNADA_ID}.pkl")
        
        logger.info("Modelos Poisson guardados")
        return True
    
    def _predict_lambdas():
        """Predice lambdas para la jornada actual"""
        import joblib
        
        logger.info("Prediciendo lambdas")
        
        # Cargar modelos
        model_H = joblib.load(f"models/poisson/model_H_{JORNADA_ID}.pkl")
        model_A = joblib.load(f"models/poisson/model_A_{JORNADA_ID}.pkl")
        ohe = joblib.load(f"models/poisson/ohe_{JORNADA_ID}.pkl")
        
        # Cargar features
        df = pd.read_feather(f"data/processed/match_features_{JORNADA_ID}.feather")
        
        # Predecir
        df_lambda = predecir_lambdas(df, model_H, model_A, ohe)
        
        # Validar lambdas
        assert (df_lambda[['lambda1', 'lambda2']] > 0).all().all(), "Lambdas negativos"
        assert (df_lambda['lambda3'] >= 0).all(), "Lambda3 negativo"
        assert (df_lambda['lambda3'] <= 1).all(), "Lambda3 > 1"
        
        # Guardar
        guardar_lambda(df_lambda, f"data/processed/lambdas_{JORNADA_ID}.csv")
        
        logger.info(f"Lambdas predichos: λ1={df_lambda['lambda1'].mean():.2f}, "
                   f"λ2={df_lambda['lambda2'].mean():.2f}, λ3={df_lambda['lambda3'].mean():.3f}")
        
        return True
    
    train_poisson = PythonOperator(
        task_id='train_poisson',
        python_callable=_train_poisson,
        dag=dag
    )
    
    predict_lambdas = PythonOperator(
        task_id='predict_lambdas',
        python_callable=_predict_lambdas,
        dag=dag
    )
    
    train_poisson >> predict_lambdas

# Stacking de probabilidades
def _stack_probabilities():
    """Combina probabilidades de mercado y Poisson"""
    logger.info("Iniciando stacking de probabilidades")
    
    # Cargar datos
    df_features = pd.read_feather(f"data/processed/match_features_{JORNADA_ID}.feather")
    df_lambda = pd.read_csv(f"data/processed/lambdas_{JORNADA_ID}.csv")
    
    # Stack con pesos óptimos
    df_blend = stack_probabilities(df_features, df_lambda, w_raw=0.58, w_poisson=0.42)
    
    # Validar
    validar_probabilidades(df_blend, cols_prefix='p_blend')
    
    # Guardar
    guardar_blend(df_blend, f"data/processed/prob_blend_{JORNADA_ID}.csv")
    
    logger.info("Stacking completado")
    return True

stack_probs = PythonOperator(
    task_id='stack_probabilities',
    python_callable=_stack_probabilities,
    dag=dag
)

# Ajuste Bayesiano
with TaskGroup('bayesian_calibration', dag=dag) as bayes_group:
    
    def _load_or_estimate_coefficients():
        """Carga o estima coeficientes bayesianos"""
        coef_file = f"models/bayes/coef_{JORNADA_ID}.json"
        
        if os.path.exists(coef_file):
            with open(coef_file, 'r') as f:
                coef = json.load(f)
            logger.info("Coeficientes cargados desde archivo")
        else:
            # Usar defaults de la metodología
            coef = {
                "k1_L": 0.15, "k2_L": -0.12, "k3_L": 0.08,
                "k1_E": -0.10, "k2_E": 0.15, "k3_E": 0.03,
                "k1_V": -0.08, "k2_V": -0.10, "k3_V": -0.05,
            }
            
            # Crear directorio si no existe
            os.makedirs("models/bayes", exist_ok=True)
            
            # Guardar para futuro
            with open(coef_file, 'w') as f:
                json.dump(coef, f, indent=2)
            
            logger.info("Usando coeficientes default")
        
        return coef
    
    def _apply_bayesian():
        """Aplica ajuste bayesiano"""
        logger.info("Aplicando calibración bayesiana")
        
        # Cargar datos
        df_blend = pd.read_csv(f"data/processed/prob_blend_{JORNADA_ID}.csv")
        df_features = pd.read_feather(f"data/processed/match_features_{JORNADA_ID}.feather")
        
        # Cargar coeficientes
        coef = _load_or_estimate_coefficients()
        
        # Aplicar ajuste
        df_final = ajustar_bayes(df_blend, df_features, coef)
        
        # Validar
        validar_probabilidades(df_final, cols_prefix='p_final')
        
        # Verificar rangos globales de la metodología
        total_L = df_final['p_final_L'].sum()
        total_E = df_final['p_final_E'].sum()
        total_V = df_final['p_final_V'].sum()
        
        logger.info(f"Totales: L={total_L:.2f}, E={total_E:.2f}, V={total_V:.2f}")
        
        # Proyección al simplex si es necesario
        if not (5.0 <= total_L <= 5.8 and 3.5 <= total_E <= 4.6 and 4.2 <= total_V <= 5.2):
            logger.warning("Proyectando al simplex por violación de rangos")
            # Implementar proyección (simplificado aquí)
            factor = 14 / (total_L + total_E + total_V)
            df_final[['p_final_L', 'p_final_E', 'p_final_V']] *= factor
        
        # Guardar
        guardar_final(df_final, f"data/processed/prob_final_{JORNADA_ID}.csv")
        
        logger.info("Ajuste bayesiano completado")
        return True
    
    apply_bayes = PythonOperator(
        task_id='apply_bayesian',
        python_callable=_apply_bayesian,
        dag=dag
    )

# Draw propensity
def _apply_draw_rules():
    """Aplica reglas especiales para empates"""
    logger.info("Aplicando draw propensity rules")
    
    # Cargar probabilidades finales
    df_final = pd.read_csv(f"data/processed/prob_final_{JORNADA_ID}.csv")
    
    # Aplicar regla
    df_draw = aplicar_draw_propensity(df_final)
    
    # Validar
    validar_probabilidades(df_draw, cols_prefix='p_final')
    
    # Estadísticas
    draw_adjusted = (
        (np.abs(df_draw['p_final_L'] - df_draw['p_final_V']) < 0.08) &
        (df_draw['p_final_E'] > df_draw[['p_final_L', 'p_final_V']].max(axis=1))
    ).sum()
    
    logger.info(f"Partidos con draw propensity: {draw_adjusted}")
    
    # Guardar
    guardar_prob_draw(df_draw, f"data/processed/prob_draw_adjusted_{JORNADA_ID}.csv")
    
    return True

draw_propensity = PythonOperator(
    task_id='apply_draw_propensity',
    python_callable=_apply_draw_rules,
    dag=dag
)

# Validación del modelo
def _validate_model_outputs():
    """Valida todos los outputs del modelo"""
    logger.info("Validando outputs del modelo")
    
    # Cargar probabilidades finales
    df = pd.read_csv(f"data/processed/prob_draw_adjusted_{JORNADA_ID}.csv")
    
    # Verificar integridad
    assert len(df) == 14, f"Se esperaban 14 partidos, hay {len(df)}"
    assert df['match_id'].nunique() == 14, "IDs duplicados"
    
    # Verificar probabilidades
    validar_probabilidades(df, cols_prefix='p_final')
    
    # Calcular métricas
    metrics = {
        'avg_p_max': df[['p_final_L', 'p_final_E', 'p_final_V']].max(axis=1).mean(),
        'n_favoritos_claros': (df[['p_final_L', 'p_final_E', 'p_final_V']].max(axis=1) > 0.60).sum(),
        'n_partidos_cerrados': (df[['p_final_L', 'p_final_E', 'p_final_V']].max(axis=1) < 0.45).sum(),
        'empates_esperados': df['p_final_E'].sum(),
        'locales_esperados': df['p_final_L'].sum(),
        'visitantes_esperados': df['p_final_V'].sum()
    }
    
    # Guardar métricas
    with open(f"data/processed/model_metrics_{JORNADA_ID}.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Métricas del modelo: {metrics}")
    logger.info("✅ Validación del modelo exitosa")
    
    return True

validate_model = PythonOperator(
    task_id='validate_model',
    python_callable=_validate_model_outputs,
    dag=dag
)

# Notificación
def _notify_completion():
    """Notifica que el modelo está listo"""
    logger.info("Modelo entrenado y calibrado exitosamente")
    
    # En producción, aquí enviarías email/Slack
    # Por ahora solo log
    
    with open(f"data/processed/model_ready_{JORNADA_ID}.flag", 'w') as f:
        f.write(str(datetime.now()))
    
    return True

notify = PythonOperator(
    task_id='notify_completion',
    python_callable=_notify_completion,
    dag=dag
)

end = DummyOperator(task_id='end', dag=dag)

# Definir flujo
start >> wait_for_features >> poisson_group >> stack_probs >> bayes_group >> draw_propensity >> validate_model >> notify >> end