import pandas as pd
import numpy as np

def generar_core_base(df_probs, df_tags):
    df = df_probs.merge(df_tags, on='match_id', how='inner')
    df = df.sort_values('match_id').reset_index(drop=True)

    signos = []
    empates_idx = []

    for i, row in df.iterrows():
        if row['etiqueta'] == 'Ancla':
            signo = row['signo_argmax']
        elif row['etiqueta'] == 'TendenciaX':
            signo = 'E'
            empates_idx.append(i)
        else:
            signo = row['signo_argmax']
            if signo == 'E':
                empates_idx.append(i)
        signos.append(signo)

    # Asegurar 4–6 empates: ajustar si necesario
    if signos.count('E') < 4:
        dif = 4 - signos.count('E')
        candidatos = df[df['signo_argmax'] != 'E'].sort_values('p_final_E', ascending=False)
        for idx in candidatos.index[:dif]:
            signos[idx] = 'E'
            empates_idx.append(idx)

    elif signos.count('E') > 6:
        dif = signos.count('E') - 6
        empates = [i for i in range(len(signos)) if signos[i] == 'E']
        for i in empates[-dif:]:
            max_signo = df.loc[i, ['p_final_L', 'p_final_V']].idxmax()[-1]
            signos[i] = max_signo
            empates_idx.remove(i)

    return signos, empates_idx

def generar_variaciones_core(base_signos, empates_idx):
    core_list = [base_signos.copy()]
    n = len(empates_idx)
    for shift in range(1, 4):
        signos_alt = base_signos.copy()
        for i in range(min(3, n)):
            idx = empates_idx[(i + shift) % n]
            signos_alt[idx] = 'L' if base_signos[idx] != 'L' else 'V'
        core_list.append(signos_alt)
    return core_list

def exportar_core(quinielas, output_path):
    df = pd.DataFrame(quinielas, columns=[f"P{i+1}" for i in range(14)])
    df.insert(0, 'quiniela_id', [f"Core-{i+1}" for i in range(len(quinielas))])
    df.to_csv(output_path, index=False)

# Ejemplo
if __name__ == "__main__":
    df_probs = pd.read_csv("data/processed/prob_draw_adjusted_2283.csv")
    df_tags = pd.read_csv("data/processed/match_tags_2283.csv")

    base_signos, empates_idx = generar_core_base(df_probs, df_tags)
    core_quinielas = generar_variaciones_core(base_signos, empates_idx)
    exportar_core(core_quinielas, "data/processed/core_quinielas_2283.csv")
