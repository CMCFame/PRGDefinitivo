import pandas as pd
import numpy as np
from scipy.stats import poisson

def bivariate_poisson_probs(lambda1, lambda2, lambda3, max_goals=6):
    p_L, p_E, p_V = [], [], []
    for l1, l2, l3 in zip(lambda1, lambda2, lambda3):
        prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
        for x in range(max_goals + 1):
            for y in range(max_goals + 1):
                min_xy = min(x, y)
                coef = np.exp(-(l1 + l2 + l3)) * l1**x * l2**y / (np.math.factorial(x) * np.math.factorial(y))
                sum_term = sum([l3**k / np.math.factorial(k) *
                                np.math.comb(x, k) * np.math.comb(y, k)
                                for k in range(min_xy + 1)])
                prob_matrix[x, y] = coef * sum_term

        p_L.append(np.tril(prob_matrix, -1).sum())  # x > y
        p_E.append(np.trace(prob_matrix))          # x = y
        p_V.append(np.triu(prob_matrix, 1).sum())  # x < y

    return np.array(p_L), np.array(p_E), np.array(p_V)

def stack_probabilities(df_features, df_lambda, w_raw=0.58, w_poisson=0.42):
    df = df_features.merge(df_lambda, on='match_id', how='inner')
    pL_p, pE_p, pV_p = bivariate_poisson_probs(df['lambda1'], df['lambda2'], df['lambda3'])

    df['p_pois_L'] = pL_p
    df['p_pois_E'] = pE_p
    df['p_pois_V'] = pV_p

    for col in ['L', 'E', 'V']:
        df[f'p_blend_{col}'] = (
            w_raw * df[f'p_raw_{col}'] + w_poisson * df[f'p_pois_{col}']
        )

    return df[['match_id', 'p_blend_L', 'p_blend_E', 'p_blend_V']]

def guardar_blend(df, output_path):
    df.to_csv(output_path, index=False)

# Ejemplo
if __name__ == "__main__":
    df_feat = pd.read_feather("data/processed/match_features_2283.feather")
    df_lambda = pd.read_csv("data/processed/lambdas_2283.csv")

    df_blend = stack_probabilities(df_feat, df_lambda, w_raw=0.58, w_poisson=0.42)
    guardar_blend(df_blend, "data/processed/prob_blend_2283.csv")
