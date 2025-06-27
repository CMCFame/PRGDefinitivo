import pandas as pd
import numpy as np
import json
from datetime import timedelta

def normalizar_momios(df_odds):
    inv_sum = 1 / df_odds[['odds_L', 'odds_E', 'odds_V']]
    row_sum = inv_sum.sum(axis=1)
    for col in ['L', 'E', 'V']:
        df_odds[f'p_raw_{col}'] = (1 / df_odds[f'odds_{col}']) / row_sum
    return df_odds

def merge_fuentes(df_progol, df_odds, previas_json, df_elo=None, df_squad=None):
    df_previas = pd.DataFrame(previas_json)

    df = df_progol.merge(df_odds, on=['home', 'away', 'fecha'], how='left')
    df = df.merge(df_previas, left_on='match_id', right_on='match_id', how='left')

    if df_elo is not None:
        df = df.merge(df_elo, on=['home', 'away', 'fecha'], how='left')
    if df_squad is not None:
        df = df.merge(df_squad, on=['home', 'away'], how='left')

    return df

def construir_features(df):
    # Diferenciales de forma (conteo de GF-GA implícito)
    df['gf5_H'] = df['form_H'].str.count('W') * 3 + df['form_H'].str.count('E')
    df['gf5_A'] = df['form_A'].str.count('W') * 3 + df['form_A'].str.count('E')
    df['delta_forma'] = df['gf5_H'] - df['gf5_A']

    # H2H ratio
    h_sum = df['h2h_H'] + df['h2h_E'] + df['h2h_A']
    df['h2h_ratio'] = (df['h2h_H'] - df['h2h_A']) / h_sum.replace(0, np.nan)

    # Diferencial Elo
    df['elo_diff'] = df['elo_home'] - df['elo_away']

    # Lesiones ponderadas
    df['inj_weight'] = df[['inj_H', 'inj_A']].sum(axis=1) / 11

    # Flags contextuales
    df['is_final'] = df['context_flag'].apply(lambda x: 'final' in x if isinstance(x, list) else False)
    df['is_derby'] = df['context_flag'].apply(lambda x: 'derbi' in x if isinstance(x, list) else False)

    # Draw Propensity Flag
    df['draw_propensity_flag'] = (
        (np.abs(df['p_raw_L'] - df['p_raw_V']) < 0.08) &
        (df['p_raw_E'] > df[['p_raw_L', 'p_raw_V']].max(axis=1))
    )

    return df

def guardar_features(df, output_path):
    df.to_feather(output_path)

# Ejemplo de uso
if __name__ == "__main__":
    df_progol = pd.read_csv("data/raw/Progol.csv", parse_dates=["fecha"])
    df_odds = pd.read_csv("data/raw/odds.csv", parse_dates=["fecha"])
    with open("data/json_previas/previas_2283.json", "r", encoding="utf-8") as f:
        previas_json = json.load(f)

    df_odds = normalizar_momios(df_odds)
    df_elo = pd.read_csv("data/raw/elo.csv") if 'elo.csv' in open("data/raw").read() else None
    df_squad = pd.read_csv("data/raw/squad_value.csv") if 'squad_value.csv' in open("data/raw").read() else None

    df_merged = merge_fuentes(df_progol, df_odds, previas_json, df_elo, df_squad)
    df_final = construir_features(df_merged)

    guardar_features(df_final, "data/processed/match_features_2283.feather")
