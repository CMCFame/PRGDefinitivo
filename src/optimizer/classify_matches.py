import pandas as pd
import numpy as np

def etiquetar_partidos(df_probs, df_features):
    df = df_probs.merge(df_features, on='match_id', how='left')

    # Probabilidad máxima y signo asociado
    df['p_max'] = df[['p_final_L', 'p_final_E', 'p_final_V']].max(axis=1)
    df['signo_argmax'] = df[['p_final_L', 'p_final_E', 'p_final_V']].idxmax(axis=1).str[-1]

    condiciones = [
        (df['p_max'] > 0.60),
        ((df['p_max'] > 0.40) & (df['p_max'] < 0.60)) | (df.get('volatilidad_flag', False) == True),
        (df.get('draw_propensity_flag', False) == True)
    ]
    etiquetas = ['Ancla', 'Divisor', 'TendenciaX']

    df['etiqueta'] = np.select(condiciones, etiquetas, default='Neutro')

    return df[['match_id', 'etiqueta', 'p_max', 'signo_argmax']]

def guardar_etiquetas(df, output_path):
    df.to_csv(output_path, index=False)

# Ejemplo
if __name__ == "__main__":
    df_probs = pd.read_csv("data/processed/prob_draw_adjusted_2283.csv")
    df_feat = pd.read_feather("data/processed/match_features_2283.feather")

    df_tags = etiquetar_partidos(df_probs, df_feat)
    guardar_etiquetas(df_tags, "data/processed/match_tags_2283.csv")
