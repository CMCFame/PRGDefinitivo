import os
from dotenv import load_dotenv

load_dotenv()

def get_env(key: str, default=None, cast=str):
    val = os.getenv(key, default)
    return cast(val) if val is not None else None

# === Paths ===
RAW_PATH = get_env("DATA_RAW_PATH", "data/raw/")
PROC_PATH = get_env("DATA_PROCESSED_PATH", "data/processed/")
JSON_PATH = get_env("JSON_PREVIAS_PATH", "data/json_previas/")

# === Parámetros operativos ===
JORNADA_ID = int(get_env("JORNADA_ID", 2283))
N_QUINIELAS = int(get_env("N_QUINIELAS", 30))
COSTO_BOLETO = float(get_env("COSTO_BOLETO", 15))
PREMIO_CAT2 = float(get_env("PREMIO_CAT2", 90000))
N_MONTECARLO = int(get_env("N_MONTECARLO_SAMPLES", 50000))

# === Archivos clave ===
ARCH_PREVIAS = f"{RAW_PATH}{get_env('PREVIAS_PDF')}"
ARCH_PROGOL = f"{RAW_PATH}{get_env('PROGOL_CSV')}"
ARCH_ODDS = f"{RAW_PATH}{get_env('ODDS_CSV')}"
ARCH_ELO = f"{RAW_PATH}{get_env('ELO_CSV')}"
ARCH_SQUAD = f"{RAW_PATH}{get_env('SQUAD_CSV')}"

# === Flags ===
ALERTAS = get_env("ENABLE_ALERTS", "false").lower() == "true"
