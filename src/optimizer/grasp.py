import pandas as pd
import numpy as np
import random
from itertools import combinations

def estimar_pr11(prob_vector, n_samples=20000):
    acc = np.random.binomial(1, prob_vector.repeat(n_samples).reshape(-1, 14))
    hits = acc.sum(axis=1)
    return (hits >= 11).mean()

def generar_candidatos(base, n=500):
    candidatos = []
    signos = ['L', 'E', 'V']
    for _ in range(n):
        nueva = base.copy()
        idxs = random.sample(range(14), random.randint(6, 10))
        for i in idxs:
            nueva[i] = random.choice([s for s in signos if s != base[i]])
        candidatos.append(nueva)
    return candidatos

def grasp_portfolio(core_df, sat_df, df_prob, alpha=0.15, n_target=30):
    df_all = pd.concat([core_df, sat_df], ignore_index=True)
    port = df_all.drop(columns='quiniela_id').values.tolist()
    port_ids = df_all['quiniela_id'].tolist()

    base = port[0]
    candidatos = generar_candidatos(base, n=500)

    while len(port) < n_target:
        marginal = []
        for q in candidatos:
            # Para cada quiniela, obtenemos vector de prob acierto
            s = []
            for i, signo in enumerate(q):
                p = df_prob.loc[i, f"p_final_{signo}"]
                s.append(p)
            pr11 = estimar_pr11(np.array(s))
            marginal.append((q, pr11))

        # Ordenar por beneficio
        marginal.sort(key=lambda x: -x[1])
        top_k = int(len(marginal) * alpha)
        elegido = random.choice(marginal[:top_k])
        port.append(elegido[0])
        port_ids.append(f"GRASP-{len(port_ids) + 1}")

    df = pd.DataFrame(port, columns=[f"P{i+1}" for i in range(14)])
    df.insert(0, 'quiniela_id', port_ids)
    return df

def exportar_grasp(df, output_path):
    df.to_csv(output_path, index=False)

# Ejemplo
if __name__ == "__main__":
    df_core = pd.read_csv("data/processed/core_quinielas_2283.csv")
    df_sat = pd.read_csv("data/processed/satellite_quinielas_2283.csv")
    df_prob = pd.read_csv("data/processed/prob_draw_adjusted_2283.csv")

    df_port = grasp_portfolio(df_core, df_sat, df_prob)
    exportar_grasp(df_port, "data/processed/portfolio_preanneal_2283.csv")
