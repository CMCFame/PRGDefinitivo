import pandas as pd

def validar_progol(df):
    assert 'match_id' in df.columns, "Falta match_id"
    assert df['match_id'].is_unique, "match_id duplicado"
    assert df[['home', 'away', 'fecha']].notnull().all().all(), "Campos clave nulos"
    return True

def validar_momios(df):
    for col in ['odds_L', 'odds_E', 'odds_V']:
        assert (df[col] > 1.01).all(), f"Momio inválido en {col}"
    return True

def validar_probabilidades(df, cols_prefix='p_final'):
    cols = [f"{cols_prefix}_{x}" for x in ['L', 'E', 'V']]
    df['p_sum'] = df[cols].sum(axis=1)
    assert (df['p_sum'].sub(1).abs() < 1e-4).all(), "Probabilidades no normalizadas"
    return True

def validar_quinielas(df):
    signos = {'L', 'E', 'V'}
    for i in range(1, 15):
        assert df[f"P{i}"].isin(signos).all(), f"Signo inválido en P{i}"
    return True
