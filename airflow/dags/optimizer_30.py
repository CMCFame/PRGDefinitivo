from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.task_group import TaskGroup
from airflow.sensors.filesystem import FileSensor
from datetime import datetime, timedelta
import os
import sys
import time
import pandas as pd
import numpy as np

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src.optimizer.classify_matches import etiquetar_partidos, guardar_etiquetas
from src.optimizer.generate_core import (generar_core_base, generar_variaciones_core, 
                                         exportar_core)
from src.optimizer.generate_satellites import generar_satellites, exportar_satellites
from src.optimizer.grasp import grasp_portfolio, exportar_grasp
from src.optimizer.annealing import annealing_optimize, exportar_portafolio_final
from src.optimizer.checklist import check_portfolio
from src.simulation.montecarlo_sim import simular_portafolio
from src.api.export_portfolio import calcular_sha, exportar
from src.utils.logger import get_logger
from src.utils.config import JORNADA_ID, N_QUINIELAS

logger = get_logger("optimizer_30_dag")

default_args = {
    'owner': 'progol',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'optimizer_30_portfolio',
    default_args=default_args,
    description='Pipeline de optimización para generar 30 quinielas',
    schedule_interval='0 1 * * *',  # 1:30 AM diario
    catchup=False,
    tags=['optimizer', 'portfolio', 'progol']
)

# Sensor para esperar modelo
wait_for_model = FileSensor(
    task_id='wait_for_model',
    filepath=f"data/processed/prob_draw_adjusted_{JORNADA_ID}.csv",
    poke_interval=30,
    timeout=600,
    dag=dag
)

start = DummyOperator(task_id='start', dag=dag)

# Clasificación de partidos
def _classify_matches():
    """Etiqueta partidos según su perfil"""
    logger.info("Clasificando partidos")
    
    # Cargar datos
    df_probs = pd.read_csv(f"data/processed/prob_draw_adjusted_{JORNADA_ID}.csv")
    df_features = pd.read_feather(f"data/processed/match_features_{JORNADA_ID}.feather")
    
    # Etiquetar
    df_tags = etiquetar_partidos(df_probs, df_features)
    
    # Estadísticas
    stats = df_tags['etiqueta'].value_counts().to_dict()
    logger.info(f"Distribución de etiquetas: {stats}")
    
    # Validar mínimos
    if stats.get('Ancla', 0) < 2:
        logger.warning("Pocos partidos Ancla, puede afectar estabilidad")
    
    if stats.get('Divisor', 0) < 4:
        logger.warning("Pocos Divisores, reduciendo diversificación")
    
    # Guardar
    guardar_etiquetas(df_tags, f"data/processed/match_tags_{JORNADA_ID}.csv")
    
    return True

classify = PythonOperator(
    task_id='classify_matches',
    python_callable=_classify_matches,
    dag=dag
)

# Generación de Core
with TaskGroup('core_generation', dag=dag) as core_group:
    
    def _generate_core():
        """Genera las 4 quinielas Core"""
        logger.info("Generando quinielas Core")
        start_time = time.time()
        
        # Cargar datos
        df_probs = pd.read_csv(f"data/processed/prob_draw_adjusted_{JORNADA_ID}.csv")
        df_tags = pd.read_csv(f"data/processed/match_tags_{JORNADA_ID}.csv")
        
        # Generar base
        base_signos, empates_idx = generar_core_base(df_probs, df_tags)
        
        logger.info(f"Core base: {' '.join(base_signos)}")
        logger.info(f"Empates en posiciones: {empates_idx}")
        
        # Generar 4 variaciones
        core_quinielas = generar_variaciones_core(base_signos, empates_idx)
        
        # Validar cada Core
        for i, q in enumerate(core_quinielas):
            empates = q.count('E')
            locales = q.count('L')
            visitantes = q.count('V')
            
            assert 4 <= empates <= 6, f"Core {i+1} con {empates} empates"
            assert 5 <= locales <= 6, f"Core {i+1} con {locales} locales"
            
            logger.info(f"Core {i+1}: L={locales} E={empates} V={visitantes}")
        
        # Exportar
        exportar_core(core_quinielas, f"data/processed/core_quinielas_{JORNADA_ID}.csv")
        
        elapsed = time.time() - start_time
        logger.info(f"Core generado en {elapsed:.2f} segundos")
        
        return True
    
    generate_core = PythonOperator(
        task_id='generate_core',
        python_callable=_generate_core,
        dag=dag
    )

# Generación de Satélites
def _generate_satellites():
    """Genera los 26 satélites en pares"""
    logger.info("Generando quinielas Satélite")
    
    # Cargar datos
    df_core = pd.read_csv(f"data/processed/core_quinielas_{JORNADA_ID}.csv")
    df_tags = pd.read_csv(f"data/processed/match_tags_{JORNADA_ID}.csv")
    
    # Usar el primer Core como base
    core_base = df_core.iloc[0].drop('quiniela_id').tolist()
    
    # Generar satélites
    sat_quinielas = generar_satellites(df_tags, core_base)
    
    # Validar correlación negativa
    correlations = []
    for i in range(0, len(sat_quinielas), 2):
        if i+1 < len(sat_quinielas):
            q1 = sat_quinielas[i][1]
            q2 = sat_quinielas[i+1][1]
            
            # Convertir a números para correlación
            v1 = [1 if s=='L' else 0 if s=='E' else -1 for s in q1]
            v2 = [1 if s=='L' else 0 if s=='E' else -1 for s in q2]
            
            corr = np.corrcoef(v1, v2)[0,1]
            correlations.append(corr)
            
            logger.info(f"Par {i//2 + 1}: correlación = {corr:.3f}")
    
    avg_corr = np.mean(correlations)
    logger.info(f"Correlación promedio pares: {avg_corr:.3f}")
    
    if avg_corr > -0.20:
        logger.warning("Correlación insuficientemente negativa")
    
    # Exportar
    exportar_satellites(sat_quinielas, f"data/processed/satellite_quinielas_{JORNADA_ID}.csv")
    
    logger.info(f"Generados {len(sat_quinielas)} satélites")
    
    return True

generate_satellites = PythonOperator(
    task_id='generate_satellites',
    python_callable=_generate_satellites,
    dag=dag
)

# GRASP
def _run_grasp():
    """Ejecuta algoritmo GRASP para completar a 30"""
    logger.info("Iniciando GRASP")
    start_time = time.time()
    
    # Cargar datos
    df_core = pd.read_csv(f"data/processed/core_quinielas_{JORNADA_ID}.csv")
    df_sat = pd.read_csv(f"data/processed/satellite_quinielas_{JORNADA_ID}.csv")
    df_prob = pd.read_csv(f"data/processed/prob_draw_adjusted_{JORNADA_ID}.csv")
    
    # Verificar que tenemos suficientes quinielas base
    n_existing = len(df_core) + len(df_sat)
    logger.info(f"Quinielas existentes: {n_existing}")
    
    if n_existing >= N_QUINIELAS:
        logger.info("Ya tenemos suficientes quinielas")
        # Solo concatenar
        df_port = pd.concat([df_core, df_sat], ignore_index=True).head(N_QUINIELAS)
    else:
        # Ejecutar GRASP
        df_port = grasp_portfolio(df_core, df_sat, df_prob, alpha=0.15, n_target=N_QUINIELAS)
    
    # Validar unicidad
    quinielas_str = df_port.drop(columns='quiniela_id').apply(lambda x: ''.join(x), axis=1)
    n_unique = quinielas_str.nunique()
    
    if n_unique < N_QUINIELAS:
        logger.error(f"Solo {n_unique} quinielas únicas de {N_QUINIELAS}")
        raise ValueError("Quinielas duplicadas detectadas")
    
    # Exportar
    exportar_grasp(df_port, f"data/processed/portfolio_preanneal_{JORNADA_ID}.csv")
    
    elapsed = time.time() - start_time
    logger.info(f"GRASP completado en {elapsed:.2f} segundos")
    
    return True

grasp = PythonOperator(
    task_id='run_grasp',
    python_callable=_run_grasp,
    dag=dag
)

# Annealing
def _run_annealing():
    """Optimización final con Simulated Annealing"""
    logger.info("Iniciando Simulated Annealing")
    start_time = time.time()
    
    # Cargar datos
    df_port = pd.read_csv(f"data/processed/portfolio_preanneal_{JORNADA_ID}.csv")
    df_prob = pd.read_csv(f"data/processed/prob_draw_adjusted_{JORNADA_ID}.csv")
    
    # Convertir a lista de quinielas
    quinielas = df_port.drop(columns='quiniela_id').values.tolist()
    
    # Calcular F inicial
    from src.optimizer.annealing import F
    f_inicial = F(quinielas, df_prob)
    logger.info(f"F inicial: {f_inicial:.4f}")
    
    # Optimizar
    port_optimizado = annealing_optimize(
        quinielas, 
        df_prob, 
        T0=0.05, 
        beta=0.92, 
        max_iter=2000
    )
    
    # Calcular F final
    f_final = F(port_optimizado, df_prob)
    logger.info(f"F final: {f_final:.4f}")
    logger.info(f"Mejora: {((f_final - f_inicial) / f_inicial * 100):.2f}%")
    
    # Exportar
    exportar_portafolio_final(port_optimizado, f"data/processed/portfolio_final_{JORNADA_ID}.csv")
    
    elapsed = time.time() - start_time
    logger.info(f"Annealing completado en {elapsed:.2f} segundos")
    
    return True

annealing = PythonOperator(
    task_id='run_annealing',
    python_callable=_run_annealing,
    dag=dag
)

# Validación con Checklist
def _validate_portfolio():
    """Valida el portafolio final contra todas las reglas"""
    logger.info("Validando portafolio final")
    
    # Cargar portafolio
    df_port = pd.read_csv(f"data/processed/portfolio_final_{JORNADA_ID}.csv")
    
    # Ejecutar checklist
    is_valid = check_portfolio(df_port)
    
    if not is_valid:
        raise ValueError("Portafolio no cumple con checklist")
    
    # Calcular estadísticas adicionales
    quinielas = df_port.drop(columns='quiniela_id').values
    
    # Distribución global
    total_L = (quinielas == 'L').sum()
    total_E = (quinielas == 'E').sum()
    total_V = (quinielas == 'V').sum()
    total = total_L + total_E + total_V
    
    logger.info(f"Distribución global: L={total_L/total:.2%}, E={total_E/total:.2%}, V={total_V/total:.2%}")
    
    # Cobertura por partido
    coverage = []
    for i in range(14):
        signos_partido = quinielas[:, i]
        unique = len(set(signos_partido))
        coverage.append(unique)
    
    logger.info(f"Cobertura promedio por partido: {np.mean(coverage):.2f} signos distintos")
    
    # Guardar estadísticas
    stats = {
        'n_quinielas': len(df_port),
        'distribucion': {'L': int(total_L), 'E': int(total_E), 'V': int(total_V)},
        'cobertura_promedio': float(np.mean(coverage)),
        'validado': True,
        'timestamp': str(datetime.now())
    }
    
    import json
    with open(f"data/processed/portfolio_stats_{JORNADA_ID}.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info("✅ Portafolio validado exitosamente")
    
    return True

validate_portfolio = PythonOperator(
    task_id='validate_portfolio',
    python_callable=_validate_portfolio,
    dag=dag
)

# Simulación Monte Carlo
def _run_simulation():
    """Simula resultados del portafolio"""
    logger.info("Ejecutando simulación Monte Carlo")
    
    # Cargar datos
    df_port = pd.read_csv(f"data/processed/portfolio_final_{JORNADA_ID}.csv")
    df_prob = pd.read_csv(f"data/processed/prob_draw_adjusted_{JORNADA_ID}.csv")
    
    # Simular
    df_sim = simular_portafolio(df_port, df_prob)
    
    # Guardar resultados detallados
    df_sim.to_csv(f"data/processed/simulation_metrics_{JORNADA_ID}.csv", index=False)
    
    # Métricas agregadas
    pr11_total = df_sim['pr_11'].mean()
    pr10_total = df_sim['pr_10'].mean()
    mu_total = df_sim['mu'].mean()
    sigma_total = df_sim['sigma'].mean()
    
    logger.info(f"Pr[≥11]: {pr11_total:.2%}")
    logger.info(f"Pr[≥10]: {pr10_total:.2%}")
    logger.info(f"μ hits: {mu_total:.2f}")
    logger.info(f"σ hits: {sigma_total:.2f}")
    
    # ROI estimado
    from src.utils.config import COSTO_BOLETO, PREMIO_CAT2
    roi = pr11_total * PREMIO_CAT2 / (N_QUINIELAS * COSTO_BOLETO) - 1
    logger.info(f"ROI estimado: {roi:.2%}")
    
    return True

simulate = PythonOperator(
    task_id='run_simulation',
    python_callable=_run_simulation,
    dag=dag
)

# Exportación final
def _export_portfolio():
    """Exporta el portafolio en formatos finales"""
    logger.info("Exportando portafolio final")
    
    # Ejecutar exportación
    exportar()
    
    # Crear resumen para impresión
    df_port = pd.read_csv(f"data/processed/portfolio_final_{JORNADA_ID}.csv")
    
    # Formato para impresión fácil
    output_lines = []
    output_lines.append(f"PORTAFOLIO PROGOL - JORNADA {JORNADA_ID}")
    output_lines.append("=" * 50)
    
    for _, row in df_port.iterrows():
        line = f"{row['quiniela_id']:>10}: {' '.join(row.drop('quiniela_id'))}"
        output_lines.append(line)
    
    # Guardar archivo de texto
    with open(f"data/processed/portfolio_print_{JORNADA_ID}.txt", 'w') as f:
        f.write('\n'.join(output_lines))
    
    logger.info("Portafolio exportado exitosamente")
    
    return True

export = PythonOperator(
    task_id='export_portfolio',
    python_callable=_export_portfolio,
    dag=dag
)

end = DummyOperator(task_id='end', dag=dag)

# Definir flujo
(start >> wait_for_model >> classify >> 
 [core_group, generate_satellites] >> 
 grasp >> annealing >> validate_portfolio >> 
 simulate >> export >> end)