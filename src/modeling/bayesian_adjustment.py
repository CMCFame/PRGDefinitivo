import pandas as pd
import numpy as np

def ajustar_bayes(df_blend, df_feat, coef=None):
    # Coeficientes por defecto si no hay entrenamiento
    coef = coef or {
        "k1_L": +0.15, "k2_L": -0.12, "k3_L": +0.08,
        "k1_E": -0.10, "k2_E": +0.15, "k3_E": +0.03,
        "k1_V": -0.08, "k2_V": -0.10, "k3_V": -0.05,
    }

    df = df_blend.merge(df_feat, on='match_id', how='left')

    # Variables
    Δf = df['delta_forma'].fillna(0)
    Ldiff = df['inj_weight'].fillna(0)
    C = (df['is_final'].astype(int) | df['is_derby'].astype(int))

    def bayes_adjust(p, k1, k2, k3):
        raw = p * (1 + k1 * Δf + k2 * Ldiff + k3 * C)
        return raw.clip(lower=1e-4)

    # Calcular y normalizar
    pL = bayes_adjust(df['p_blend_L'], coef["k1_L"], coef["k2_L"], coef["k3_L"])
    pE = bayes_adjust(df['p_blend_E'], coef["k1_E"], coef["k2_E"], coef["k3_E"])
    pV = bayes_adjust(df['p_blend_V'], coef["k1_V"], coef["k2_V"], coef["k3_V"])
    Z = pL + pE + pV

    df['p_final_L'] = (pL / Z).clip(0, 1)
    df['p_final_E'] = (pE / Z).clip(0, 1)
    df['p_final_V'] = (pV / Z).clip(0, 1)

    return df[['match_id', 'p_final_L', 'p_final_E', 'p_final_V']]

def guardar_final(df, output_path):
    df.to_csv(output_path, index=False)

# Ejemplo
if __name__ == "__main__":
    df_blend = pd.read_csv("data/processed/prob_blend_2283.csv")
    df_feat = pd.read_feather("data/processed/match_features_2283.feather")

    df_final = ajustar_bayes(df_blend, df_feat)
    guardar_final(df_final, "data/processed/prob_final_2283.csv")
