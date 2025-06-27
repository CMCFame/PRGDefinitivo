import pandas as pd
from src.utils.logger import get_logger
from src.utils.config import ARCH_PROGOL
from src.utils.validators import validar_progol

logger = get_logger("ingest_csv")

def cargar_progol(path=ARCH_PROGOL):
    logger.info(f"Cargando archivo Progol desde {path}")
    df = pd.read_csv(path, parse_dates=["fecha"])

    # Generar match_id
    df["match_id"] = df.apply(lambda r: f"{r['concurso_id']}-{r['match_no']}", axis=1)

    # Validar consistencia
    df["resultado_calc"] = df.apply(
        lambda r: "L" if r["l_g"] > r["a_g"] else "E" if r["l_g"] == r["a_g"] else "V", axis=1
    )
    inconsistentes = (df["resultado"] != df["resultado_calc"]).sum()
    if inconsistentes > 0:
        logger.warning(f"{inconsistentes} resultados recalculados por inconsistencia de goles")
        df["resultado"] = df["resultado_calc"]

    validar_progol(df)
    logger.info("Archivo Progol validado correctamente.")
    return df

if __name__ == "__main__":
    df = cargar_progol()
    df.to_csv("data/processed/progol_clean.csv", index=False)
