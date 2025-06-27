import pandas as pd
from src.utils.logger import get_logger
from src.utils.config import ARCH_ODDS
from src.utils.validators import validar_momios

logger = get_logger("scrape_odds")

def cargar_odds(path=ARCH_ODDS):
    logger.info(f"Cargando momios desde {path}")
    df = pd.read_csv(path, parse_dates=["fecha"])

    # Verificación mínima
    df["match_id"] = df.apply(lambda r: f"{r['concurso_id']}-{r['match_no']}", axis=1)
    validar_momios(df)
    logger.info("Momios cargados y validados.")
    return df

if __name__ == "__main__":
    df = cargar_odds()
    df.to_csv("data/processed/odds_clean.csv", index=False)
