import pandas as pd
import hashlib
import json
from datetime import datetime
from src.utils.logger import get_logger
from src.utils.config import PROC_PATH, JORNADA_ID

logger = get_logger("export_portfolio")

def calcular_sha(quinielas_df):
    filas_concat = quinielas_df.drop(columns="quiniela_id").astype(str).agg("".join, axis=1)
    hash_total = hashlib.sha1("".join(filas_concat).encode()).hexdigest()
    return hash_total

def exportar(archivo="portfolio_final_{}.csv".format(JORNADA_ID)):
    df = pd.read_csv(f"{PROC_PATH}{archivo}")
    hash_id = calcular_sha(df)

    df.to_csv(f"{PROC_PATH}export_quinielas_{JORNADA_ID}.csv", index=False)
    logger.info(f"CSV exportado con hash {hash_id}")

    resumen = {
        "jornada": JORNADA_ID,
        "exportado": datetime.now().isoformat(),
        "hash": hash_id,
        "n_quinielas": len(df),
        "signos_L": int((df == "L").sum().sum()),
        "signos_E": int((df == "E").sum().sum()),
        "signos_V": int((df == "V").sum().sum()),
    }

    with open(f"{PROC_PATH}export_quinielas_{JORNADA_ID}.json", "w") as f:
        json.dump(resumen, f, indent=2)
    logger.info("Resumen exportado en JSON.")

if __name__ == "__main__":
    exportar()
