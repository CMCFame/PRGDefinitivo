import pandas as pd
import numpy as np

def aplicar_draw_propensity(df):
    df = df.copy()
    cond = (
        (np.abs(df['p_final_L'] - df['p_final_V']) < 0.08) &
        (df['p_final_E'] > df[['p_final_L', 'p_final_V']].max(axis=1))
    )

    # Corrección
    df.loc[cond, 'p_final_E'] += 0.06
    df.loc[cond, 'p_final_L'] -= 0.03
    df.loc[cond, 'p_final_V'] -= 0.03

    # Renormalización (suma = 1)
    total = df[['p_final_L', 'p_final_E', 'p_final_V']].sum(axis=1)
    for col in ['p_final_L', 'p_final_E', 'p_final_V']:
        df[col] = df[col] / total

    return df[['match_id', 'p_final_L', 'p_final_E', 'p_final_V']]

def guardar_prob_draw(df, output_path):
    df.to_csv(output_path, index=False)

# Ejemplo
if __name__ == "__main__":
    df = pd.read_csv("data/processed/prob_final_2283.csv")
    df_draw = aplicar_draw_propensity(df)
    guardar_prob_draw(df_draw, "data/processed/prob_draw_adjusted_2283.csv")
