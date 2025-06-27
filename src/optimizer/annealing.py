import pandas as pd
import numpy as np
import random
import math

def pr_11_boleto(q, df_prob, n_samples=20000):
    s = [df_prob.loc[i, f"p_final_{signo}"] for i, signo in enumerate(q)]
    acc = np.random.binomial(1, np.array(s).repeat(n_samples).reshape(-1, 14))
    hits = acc.sum(axis=1)
    return (hits >= 11).mean()

def F(portfolio, df_prob):
    vals = [1 - pr_11_boleto(q, df_prob) for q in portfolio]
    prod = np.prod(vals)
    return 1 - prod

def annealing_optimize(port_init, df_prob, T0=0.05, beta=0.92, max_iter=2000):
    current = port_init.copy()
    best = current.copy()
    f_best = F(current, df_prob)
    T = T0
    stagnation = 0

    for step in range(10000):
        q_idx = random.randint(0, len(current) - 1)
        q = current[q_idx].copy()
        idxs = random.sample(range(14), random.randint(1, 3))

        q_new = q.copy()
        for i in idxs:
            original = q_new[i]
            q_new[i] = random.choice(['L', 'E', 'V'])
            while q_new[i] == original:
                q_new[i] = random.choice(['L', 'E', 'V'])

        candidate = current.copy()
        candidate[q_idx] = q_new
        f_cand = F(candidate, df_prob)

        delta = f_cand - f_best
        if delta > 0 or random.random() < math.exp(delta / T):
            current = candidate
            if f_cand > f_best:
                best = candidate
                f_best = f_cand
                stagnation = 0
            else:
                stagnation += 1
        else:
            stagnation += 1

        T *= beta
        if stagnation > max_iter:
            break

    return best

def exportar_portafolio_final(portfolio, output_path):
    df = pd.DataFrame(portfolio, columns=[f"P{i+1}" for i in range(14)])
    df.insert(0, 'quiniela_id', [f"Q{i+1}" for i in range(len(portfolio))])
    df.to_csv(output_path, index=False)

# Ejemplo
if __name__ == "__main__":
    df_port = pd.read_csv("data/processed/portfolio_preanneal_2283.csv")
    df_prob = pd.read_csv("data/processed/prob_draw_adjusted_2283.csv")

    quinielas = df_port.drop(columns='quiniela_id').values.tolist()
    port_final = annealing_optimize(quinielas, df_prob)

    exportar_portafolio_final(port_final, "data/processed/portfolio_final_2283.csv")
