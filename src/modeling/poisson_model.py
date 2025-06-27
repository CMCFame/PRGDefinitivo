import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

def preparar_dataset_poisson(df):
    df = df.copy()
    df['elo_diff'] = df['elo_home'] - df['elo_away']
    df['log_lambda_1'] = np.log1p(df['goles_H'])
    df['log_lambda_2'] = np.log1p(df['goles_A'])
    return df

def entrenar_poisson_model(df):
    ohe = OneHotEncoder(drop='first', sparse=False)

    # One-hot para equipos y ligas
    X_equipos = ohe.fit_transform(df[['home', 'away', 'liga']])
    col_names = ohe.get_feature_names_out(['home', 'away', 'liga'])
    X = pd.DataFrame(X_equipos, columns=col_names, index=df.index)

    # Features continuas
    X['elo_diff'] = df['elo_diff']
    X['factor_local'] = df['factor_local']

    # Targets
    y_H = df['goles_H']
    y_A = df['goles_A']

    # Ridge GLM (Poisson → usando MSE como proxy con log transform)
    model_H = Ridge(alpha=0.005)
    model_A = Ridge(alpha=0.005)

    model_H.fit(X, y_H)
    model_A.fit(X, y_A)

    return model_H, model_A, ohe

def predecir_lambdas(df, model_H, model_A, ohe):
    X_equipos = ohe.transform(df[['home', 'away', 'liga']])
    X = pd.DataFrame(X_equipos, columns=ohe.get_feature_names_out(['home', 'away', 'liga']))
    X['elo_diff'] = df['elo_home'] - df['elo_away']
    X['factor_local'] = df['factor_local']

    λ1 = model_H.predict(X).clip(min=0.05)
    λ2 = model_A.predict(X).clip(min=0.05)

    # Estimar λ3 por método de momentos
    resid_H = df['goles_H'] - λ1
    resid_A = df['goles_A'] - λ2
    cov = np.clip((resid_H * resid_A).mean(), 0, 1)  # truncado como sugiere metodología

    df_out = df[['match_id']].copy()
    df_out['lambda1'] = λ1
    df_out['lambda2'] = λ2
    df_out['lambda3'] = cov  # constante por jornada, puede ser afinado
    return df_out

def guardar_lambda(df, output_path):
    df.to_csv(output_path, index=False)

# Ejemplo
if __name__ == "__main__":
    df = pd.read_feather("data/processed/match_features_2283.feather")
    df = preparar_dataset_poisson(df)
    model_H, model_A, ohe = entrenar_poisson_model(df)
    df_lambda = predecir_lambdas(df, model_H, model_A, ohe)
    guardar_lambda(df_lambda, "data/processed/lambdas_2283.csv")
