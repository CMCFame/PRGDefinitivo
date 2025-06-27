import pandas as pd
import numpy as np

def invertir_signo(signo):
    return {'L': 'V', 'V': 'L', 'E': 'E'}.get(signo, signo)

def generar_satellites(df_tags, core_base):
    divisores = df_tags[df_tags['etiqueta'] == 'Divisor']
    sat_quinielas = []

    for i, row in divisores.iterrows():
        match_idx = i
        q_plus = core_base.copy()
        q_minus = core_base.copy()

        # Asignar inversion
        q_minus[match_idx] = invertir_signo(core_base[match_idx])

        sat_quinielas.append(("Sat-{}a".format(len(sat_quinielas)//2 + 1), q_plus))
        sat_quinielas.append(("Sat-{}b".format(len(sat_quinielas)//2 + 1), q_minus))

        if len(sat_quinielas) >= 26:
            break

    return sat_quinielas

def exportar_satellites(quinielas, output_path):
    df = pd.DataFrame([q for _, q in quinielas], columns=[f"P{i+1}" for i in range(14)])
    df.insert(0, 'quiniela_id', [q[0] for q in quinielas])
    df.to_csv(output_path, index=False)

# Ejemplo
if __name__ == "__main__":
    df_core = pd.read_csv("data/processed/core_quinielas_2283.csv")
    df_tags = pd.read_csv("data/processed/match_tags_2283.csv")

    core_1 = df_core[df_core['quiniela_id'] == 'Core-1'].iloc[0].drop('quiniela_id').tolist()
    sat_quinielas = generar_satellites(df_tags, core_1)
    exportar_satellites(sat_quinielas, "data/processed/satellite_quinielas_2283.csv")
