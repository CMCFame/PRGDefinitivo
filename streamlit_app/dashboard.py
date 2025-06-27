import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def cargar_datos():
    port = pd.read_csv("data/processed/portfolio_final_2283.csv")
    prob = pd.read_csv("data/processed/prob_draw_adjusted_2283.csv")
    sim = pd.read_csv("data/processed/simulation_metrics_2283.csv")  # generado por montecarlo_sim.py
    return port, prob, sim

def mostrar_quinielas(port):
    st.subheader("📋 Quinielas Finales")
    st.dataframe(port, use_container_width=True)

def mostrar_distribucion_signos(port):
    st.subheader("📊 Distribución Global de Signos")
    signos = port.drop(columns="quiniela_id").values.flatten()
    distrib = pd.Series(signos).value_counts(normalize=True).reindex(['L','E','V'], fill_value=0)

    fig, ax = plt.subplots()
    distrib.plot(kind='bar', ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Distribución de signos en las 30 quinielas")
    st.pyplot(fig)

def mostrar_metricas(sim):
    st.subheader("📈 Métricas Probabilísticas del Portafolio")
    st.dataframe(sim.style.format("{:.3f}"), use_container_width=True)

    pr11_mean = sim["pr_11"].mean()
    st.metric(label="🔮 Pr[≥11] esperada", value=f"{pr11_mean:.2%}")

    roi_est = pr11_mean * 90000 / (30 * 15) - 1
    st.metric(label="💰 ROI estimado (MXN 90K)", value=f"{roi_est:.2%}")

def descargar_portafolio(port):
    csv = port.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Descargar CSV", csv, "quinielas_2283.csv", "text/csv")

def main():
    st.title("🔢 Dashboard de Portafolio Progol")
    port, prob, sim = cargar_datos()

    mostrar_quinielas(port)
    mostrar_distribucion_signos(port)
    mostrar_metricas(sim)
    descargar_portafolio(port)

if __name__ == "__main__":
    main()
