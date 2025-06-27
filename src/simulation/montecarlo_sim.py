import pandas as pd
import numpy as np
import ace_tools as tools

def simular_hits(q, prob_df, n_sim=50000):
    probs = [prob_df.loc[i, f"p_final_{s}"] for i, s in enumerate(q)]
    samples = np.random.binomial(1, np.tile(probs, (n_sim, 1)))
    hits = samples.sum(axis=1)
    return {
        "mu": hits.mean(),
        "sigma": hits.std(),
        "pr_10": (hits >= 10).mean(),
        "pr_11": (hits >= 11).mean()
    }

def simular_portafolio(df_port, df_prob):
    resultados = []
    for _, row in df_port.iterrows():
        q_id = row['quiniela_id']
        q = row.drop('quiniela_id').tolist()
        res = simular_hits(q, df_prob)
        res["quiniela_id"] = q_id
        resultados.append(res)
    return pd.DataFrame(resultados)

# Ejemplo
if __name__ == "__main__":
    df_port = pd.read_csv("data/processed/portfolio_final_2283.csv")
    df_prob = pd.read_csv("data/processed/prob_draw_adjusted_2283.csv")

    df_result = simular_portafolio(df_port, df_prob)

    resumen = {
        "μ hits": df_result["mu"].mean(),
        "σ hits": df_result["sigma"].mean(),
        "Pr[≥10]": df_result["pr_10"].mean(),
        "Pr[≥11]": df_result["pr_11"].mean(),
        "ROI estimado (premio MXN 90k)": df_result["pr_11"].mean() * 90000 / (30 * 15) - 1
    }

    tools.display_dataframe_to_user("Métricas del Portafolio", df_result.round(3))
    print("Resumen Global:")
    for k, v in resumen.items():
        print(f"{k}: {v:.4f}")
