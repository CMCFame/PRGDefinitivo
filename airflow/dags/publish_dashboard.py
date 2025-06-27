from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime, timedelta
import os
import sys
import shutil
import json
import pandas as pd

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src.utils.logger import get_logger
from src.utils.config import JORNADA_ID, STREAMLIT_PORT, ALERTAS, DASHBOARD_REFRESH_INTERVAL

logger = get_logger("publish_dashboard_dag")

default_args = {
    'owner': 'progol',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=2)
}

dag = DAG(
    'publish_dashboard',
    default_args=default_args,
    description='Publica resultados y actualiza dashboard',
    schedule_interval='0 2 * * *',  # 2 AM diario
    catchup=False,
    tags=['dashboard', 'publish', 'progol']
)

# Sensor para esperar portafolio
wait_for_portfolio = FileSensor(
    task_id='wait_for_portfolio',
    filepath=f"data/processed/portfolio_final_{JORNADA_ID}.csv",
    poke_interval=30,
    timeout=1200,
    dag=dag
)

start = DummyOperator(task_id='start', dag=dag)

# Preparar datos para dashboard
def _prepare_dashboard_data():
    """Prepara todos los datos necesarios para el dashboard"""
    logger.info("Preparando datos para dashboard")
    
    # Crear directorio dashboard si no existe
    dashboard_dir = "data/dashboard"
    os.makedirs(dashboard_dir, exist_ok=True)
    
    # Archivos a copiar
    files_to_copy = [
        f"portfolio_final_{JORNADA_ID}.csv",
        f"prob_draw_adjusted_{JORNADA_ID}.csv",
        f"simulation_metrics_{JORNADA_ID}.csv",
        f"model_metrics_{JORNADA_ID}.json",
        f"portfolio_stats_{JORNADA_ID}.json"
    ]
    
    # Copiar archivos
    for file in files_to_copy:
        src = f"data/processed/{file}"
        dst = f"{dashboard_dir}/{file}"
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            logger.info(f"Copiado: {file}")
        else:
            logger.warning(f"No encontrado: {file}")
    
    # Crear archivo de metadata consolidado
    metadata = {
        'jornada': JORNADA_ID,
        'fecha_generacion': str(datetime.now()),
        'archivos_disponibles': []
    }
    
    # Verificar archivos y agregar metadata
    for file in os.listdir(dashboard_dir):
        if file.endswith('.csv') or file.endswith('.json'):
            metadata['archivos_disponibles'].append({
                'nombre': file,
                'tamaño': os.path.getsize(f"{dashboard_dir}/{file}"),
                'modificado': str(datetime.fromtimestamp(os.path.getmtime(f"{dashboard_dir}/{file}")))
            })
    
    # Agregar resumen de métricas
    try:
        with open(f"{dashboard_dir}/portfolio_stats_{JORNADA_ID}.json", 'r') as f:
            stats = json.load(f)
        metadata['resumen'] = stats
        
        # Cargar métricas de simulación
        df_sim = pd.read_csv(f"{dashboard_dir}/simulation_metrics_{JORNADA_ID}.csv")
        metadata['simulacion'] = {
            'pr_11_promedio': float(df_sim['pr_11'].mean()),
            'pr_10_promedio': float(df_sim['pr_10'].mean()),
            'hits_esperados': float(df_sim['mu'].mean()),
            'desviacion_estandar': float(df_sim['sigma'].mean())
        }
    except Exception as e:
        logger.error(f"Error cargando estadísticas: {e}")
    
    # Guardar metadata
    with open(f"{dashboard_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Datos preparados para dashboard")
    return True

prepare_data = PythonOperator(
    task_id='prepare_dashboard_data',
    python_callable=_prepare_dashboard_data,
    dag=dag
)

# Generar reportes
def _generate_reports():
    """Genera reportes en PDF y HTML"""
    logger.info("Generando reportes")
    
    reports_dir = "data/reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    # Cargar datos
    df_port = pd.read_csv(f"data/dashboard/portfolio_final_{JORNADA_ID}.csv")
    df_sim = pd.read_csv(f"data/dashboard/simulation_metrics_{JORNADA_ID}.csv")
    
    # Generar reporte HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reporte Progol - Jornada {JORNADA_ID}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #34495e; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .metrics {{ background-color: #ecf0f1; padding: 15px; margin: 20px 0; }}
            .quiniela {{ font-family: monospace; letter-spacing: 2px; }}
        </style>
    </head>
    <body>
        <h1>Reporte de Quinielas - Jornada {JORNADA_ID}</h1>
        <p>Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="metrics">
            <h2>Métricas del Portafolio</h2>
            <ul>
                <li>Probabilidad de ≥11 aciertos: <strong>{df_sim['pr_11'].mean():.2%}</strong></li>
                <li>Probabilidad de ≥10 aciertos: <strong>{df_sim['pr_10'].mean():.2%}</strong></li>
                <li>Aciertos esperados: <strong>{df_sim['mu'].mean():.2f}</strong></li>
                <li>Desviación estándar: <strong>{df_sim['sigma'].mean():.2f}</strong></li>
            </ul>
        </div>
        
        <h2>Portafolio de 30 Quinielas</h2>
        <table>
            <tr>
                <th>ID</th>
                <th>Quiniela</th>
                <th>Pr[≥11]</th>
                <th>μ</th>
            </tr>
    """
    
    # Agregar cada quiniela
    for idx, row in df_port.iterrows():
        quiniela_str = ' '.join(row.drop('quiniela_id'))
        sim_row = df_sim[df_sim['quiniela_id'] == row['quiniela_id']].iloc[0]
        
        html_content += f"""
            <tr>
                <td>{row['quiniela_id']}</td>
                <td class="quiniela">{quiniela_str}</td>
                <td>{sim_row['pr_11']:.2%}</td>
                <td>{sim_row['mu']:.2f}</td>
            </tr>
        """
    
    html_content += """
        </table>
    </body>
    </html>
    """
    
    # Guardar HTML
    with open(f"{reports_dir}/reporte_{JORNADA_ID}.html", 'w') as f:
        f.write(html_content)
    
    logger.info("Reporte HTML generado")
    
    # Generar archivo CSV consolidado para Excel
    consolidated = df_port.copy()
    
    # Agregar métricas de simulación
    consolidated = consolidated.merge(
        df_sim[['quiniela_id', 'pr_11', 'mu']], 
        on='quiniela_id', 
        how='left'
    )
    
    # Guardar Excel-friendly CSV
    consolidated.to_csv(f"{reports_dir}/portafolio_completo_{JORNADA_ID}.csv", index=False)
    
    logger.info("Reportes generados exitosamente")
    return True

generate_reports = PythonOperator(
    task_id='generate_reports',
    python_callable=_generate_reports,
    dag=dag
)

# Actualizar base de datos (si existe)
def _update_database():
    """Actualiza la base de datos con los resultados"""
    logger.info("Actualizando base de datos")
    
    try:
        import psycopg2
        from src.utils.config import get_env
        
        # Conexión a DB
        conn = psycopg2.connect(
            host=get_env('PG_HOST', 'localhost'),
            port=get_env('PG_PORT', 5432),
            database=get_env('PG_DB', 'progol'),
            user=get_env('PG_USER', 'progol_admin'),
            password=get_env('PG_PASSWORD', 'progol_pass')
        )
        
        cur = conn.cursor()
        
        # Crear tabla si no existe
        cur.execute("""
            CREATE TABLE IF NOT EXISTS portfolios (
                id SERIAL PRIMARY KEY,
                jornada_id INTEGER NOT NULL,
                fecha_generacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                hash_portfolio VARCHAR(40),
                pr_11 DECIMAL(5,4),
                pr_10 DECIMAL(5,4),
                mu_hits DECIMAL(4,2),
                sigma_hits DECIMAL(4,2),
                metadata JSONB
            )
        """)
        
        # Cargar datos
        df_port = pd.read_csv(f"data/processed/portfolio_final_{JORNADA_ID}.csv")
        df_sim = pd.read_csv(f"data/dashboard/simulation_metrics_{JORNADA_ID}.csv")
        
        # Calcular hash
        from src.api.export_portfolio import calcular_sha
        hash_port = calcular_sha(df_port)
        
        # Preparar metadata
        metadata = {
            'n_quinielas': len(df_port),
            'distribucion': {
                'L': int((df_port.drop(columns='quiniela_id') == 'L').sum().sum()),
                'E': int((df_port.drop(columns='quiniela_id') == 'E').sum().sum()),
                'V': int((df_port.drop(columns='quiniela_id') == 'V').sum().sum())
            }
        }
        
        # Insertar
        cur.execute("""
            INSERT INTO portfolios 
            (jornada_id, hash_portfolio, pr_11, pr_10, mu_hits, sigma_hits, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            JORNADA_ID,
            hash_port,
            float(df_sim['pr_11'].mean()),
            float(df_sim['pr_10'].mean()),
            float(df_sim['mu'].mean()),
            float(df_sim['sigma'].mean()),
            json.dumps(metadata)
        ))
        
        conn.commit()
        cur.close()
        conn.close()
        
        logger.info("Base de datos actualizada")
        
    except Exception as e:
        logger.warning(f"No se pudo actualizar BD: {e}")
        # No falla el pipeline si no hay BD
    
    return True

update_db = PythonOperator(
    task_id='update_database',
    python_callable=_update_database,
    dag=dag
)

# Limpiar cache de Streamlit
clear_cache = BashOperator(
    task_id='clear_streamlit_cache',
    bash_command="""
    find ~/.streamlit/cache -type f -delete 2>/dev/null || true
    echo "Cache limpiado"
    """,
    dag=dag
)

# Notificaciones
def _send_notifications():
    """Envía notificaciones del portafolio generado"""
    logger.info("Enviando notificaciones")
    
    if not ALERTAS:
        logger.info("Alertas deshabilitadas")
        return True
    
    try:
        # Cargar métricas
        df_sim = pd.read_csv(f"data/dashboard/simulation_metrics_{JORNADA_ID}.csv")
        pr11 = df_sim['pr_11'].mean()
        
        # Preparar mensaje
        mensaje = f"""
        ✅ Portafolio Progol Generado
        
        Jornada: {JORNADA_ID}
        Pr[≥11]: {pr11:.2%}
        Hora: {datetime.now().strftime('%H:%M')}
        
        Dashboard disponible en puerto {STREAMLIT_PORT}
        """
        
        # Aquí irían las integraciones con Slack/Email
        # Por ahora solo log
        logger.info(mensaje)
        
        # Crear archivo de notificación
        with open(f"data/reports/notificacion_{JORNADA_ID}.txt", 'w') as f:
            f.write(mensaje)
        
    except Exception as e:
        logger.error(f"Error en notificaciones: {e}")
    
    return True

notify = PythonOperator(
    task_id='send_notifications',
    python_callable=_send_notifications,
    dag=dag
)

# Verificar que Streamlit esté corriendo
check_streamlit = BashOperator(
    task_id='check_streamlit',
    bash_command=f"""
    if ! lsof -i:{STREAMLIT_PORT} > /dev/null 2>&1; then
        echo "Streamlit no está corriendo en puerto {STREAMLIT_PORT}"
        echo "Ejecutar: streamlit run streamlit_app/dashboard.py --server.port {STREAMLIT_PORT}"
    else
        echo "Streamlit activo en puerto {STREAMLIT_PORT}"
    fi
    """,
    dag=dag
)

# Crear índice de jornadas
def _update_index():
    """Actualiza el índice de jornadas procesadas"""
    logger.info("Actualizando índice de jornadas")
    
    index_file = "data/dashboard/jornadas_index.json"
    
    # Cargar índice existente o crear nuevo
    if os.path.exists(index_file):
        with open(index_file, 'r') as f:
            index = json.load(f)
    else:
        index = {'jornadas': []}
    
    # Agregar jornada actual si no existe
    jornada_entry = {
        'id': JORNADA_ID,
        'fecha_proceso': str(datetime.now()),
        'archivos': []
    }
    
    # Listar archivos de esta jornada
    for file in os.listdir("data/dashboard"):
        if str(JORNADA_ID) in file:
            jornada_entry['archivos'].append(file)
    
    # Actualizar índice (evitar duplicados)
    index['jornadas'] = [j for j in index['jornadas'] if j['id'] != JORNADA_ID]
    index['jornadas'].append(jornada_entry)
    
    # Ordenar por ID descendente
    index['jornadas'].sort(key=lambda x: x['id'], reverse=True)
    
    # Guardar
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2)
    
    logger.info(f"Índice actualizado con {len(index['jornadas'])} jornadas")
    
    return True

update_index = PythonOperator(
    task_id='update_jornadas_index',
    python_callable=_update_index,
    dag=dag
)

end = DummyOperator(task_id='end', dag=dag)

# Definir flujo
(start >> wait_for_portfolio >> prepare_data >> 
 [generate_reports, update_db] >> 
 clear_cache >> check_streamlit >> 
 notify >> update_index >> end)