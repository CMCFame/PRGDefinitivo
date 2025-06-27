import pandas as pd
import numpy as np
from itertools import combinations

def porcentaje_signo(q, signo):
    return q.count(signo) / 14

def max_sign_pct(q):
    return max(q.count('L'), q.count('E'), q.count('V')) / 14

def max_sign_pct_first3(q):
    return max(q[:3].count('L'), q[:3].count('E'), q[:3].count('V')) / 3

def jaccard_similarity(q1, q2):
    set1, set2 = set(enumerate(q1)), set(enumerate(q2))
    return len(set1 & set2) / len(set1 | set2)

def check_portfolio(df):
    passed = True
    quinielas = df.drop(columns='quiniela_id').values.tolist()

    for idx, q in enumerate(quinielas):
        if not (4 <= q.count('E') <= 6):
            print(f"[❌] Quiniela {idx+1}: empates fuera de rango.")
            passed = False
        if not (0.35 <= porcentaje_signo(q, 'L') <= 0.41):
            print(f"[❌] Quiniela {idx+1}: porcentaje L fuera de rango.")
            passed = False
        if max_sign_pct(q) > 0.70:
            print(f"[❌] Quiniela {idx+1}: concentración global > 70 %.")
            passed = False
        if max_sign_pct_first3(q) > 0.60:
            print(f"[❌] Quiniela {idx+1}: concentración en partidos 1–3 > 60 %.")
            passed = False

    seen = set()
    for idx, q in enumerate(quinielas):
        key = tuple(q)
        if key in seen:
            print(f"[❌] Quiniela {idx+1}: duplicada.")
            passed = False
        seen.add(key)

    sims = []
    for q1, q2 in combinations(quinielas, 2):
        sim = jaccard_similarity(q1, q2)
        sims.append(sim)
    if max(sims) > 0.85:
        print(f"[❌] Máxima similitud Jaccard entre pares: {max(sims):.2f}")
        passed = False

    if passed:
        print("[✅] Portafolio válido según checklist.")
    return passed

# Ejemplo
if __name__ == "__main__":
    df = pd.read_csv("data/processed/portfolio_final_2283.csv")
    result = check_portfolio(df)
