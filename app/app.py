"""
Application Streamlit — Agritech Answers.

Interface simple et visuelle permettant à un agriculteur de :
  - Mode "Prédiction" : estimer le rendement d'une culture qu'il a déjà choisie.
  - Mode "Recommandation" : obtenir un classement des cultures les plus
    rentables pour les conditions de sa parcelle.

Cette application ne contient AUCUNE logique de Machine Learning : elle ne
fait qu'interroger l'API FastAPI (voir ../api/main.py) et afficher les
résultats.
"""
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Agritech Answers — Rendement agricole", page_icon="🌾", layout="centered")


@st.cache_data(ttl=300)
def get_metadata():
    response = requests.get(f"{API_URL}/metadata", timeout=10)
    response.raise_for_status()
    return response.json()


def sidebar_context_inputs(metadata: dict) -> dict:
    st.sidebar.header("Conditions de la parcelle")

    region = st.sidebar.selectbox("Région", metadata["valid_regions"])
    soil_type = st.sidebar.selectbox("Type de sol", metadata["valid_soil_types"])
    weather_condition = st.sidebar.selectbox("Condition météo dominante", metadata["valid_weather"])

    rainfall_mm = st.sidebar.slider("Pluviométrie (mm/an)", min_value=50, max_value=1500, value=600, step=10)
    temperature_celsius = st.sidebar.slider("Température moyenne (°C)", min_value=-5, max_value=45, value=25, step=1)
    days_to_harvest = st.sidebar.slider("Jours avant récolte", min_value=30, max_value=200, value=110, step=1)

    fertilizer_used = st.sidebar.toggle("Engrais utilisé", value=True)
    irrigation_used = st.sidebar.toggle("Irrigation utilisée", value=True)

    return {
        "region": region,
        "soil_type": soil_type,
        "rainfall_mm": float(rainfall_mm),
        "temperature_celsius": float(temperature_celsius),
        "fertilizer_used": fertilizer_used,
        "irrigation_used": irrigation_used,
        "weather_condition": weather_condition,
        "days_to_harvest": int(days_to_harvest),
    }


def main():
    st.title("🌾 Agritech Answers")
    st.caption("Aide à la décision pour le choix et le rendement des cultures")

    try:
        metadata = get_metadata()
    except requests.exceptions.RequestException:
        st.error(
            f"Impossible de contacter l'API à l'adresse {API_URL}. "
            "Vérifiez qu'elle est bien démarrée (voir README du projet)."
        )
        st.stop()

    with st.expander("ℹ️ À propos du modèle utilisé"):
        st.write(f"**Type de modèle** : {metadata['model_type']}")
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE (t/ha)", f"{metadata['model_metrics']['rmse']:.3f}")
        col2.metric("MAE (t/ha)", f"{metadata['model_metrics']['mae']:.3f}")
        col3.metric("R²", f"{metadata['model_metrics']['r2']:.3f}")

    mode = st.radio(
        "Que souhaitez-vous faire ?",
        options=["🔮 Prédiction", "🏆 Recommandation"],
        horizontal=True,
    )

    context = sidebar_context_inputs(metadata)

    if mode == "🔮 Prédiction":
        st.subheader("Estimer le rendement d'une culture")
        crop = st.selectbox("Culture à évaluer", metadata["valid_crops"])

        if st.button("Estimer le rendement", type="primary"):
            payload = {**context, "crop": crop}
            with st.spinner("Calcul en cours..."):
                try:
                    response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
                    response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    st.error(f"Erreur lors de l'appel à l'API : {e}")
                    st.stop()

            result = response.json()
            st.success("Estimation réalisée")
            st.metric(
                label=f"Rendement estimé pour {result['crop']}",
                value=f"{result['predicted_yield_t_ha']} t/ha",
            )

    else:
        st.subheader("Trouver la culture la plus rentable")
        st.write(
            "L'application simule le rendement pour **toutes les cultures possibles** "
            "avec les conditions de parcelle renseignées à gauche, et les classe de la "
            "plus rentable à la moins rentable."
        )

        if st.button("Obtenir la recommandation", type="primary"):
            with st.spinner("Simulation en cours pour toutes les cultures..."):
                try:
                    response = requests.post(f"{API_URL}/recommend", json=context, timeout=15)
                    response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    st.error(f"Erreur lors de l'appel à l'API : {e}")
                    st.stop()

            result = response.json()
            recos = pd.DataFrame(result["recommendations"])

            best = recos.iloc[0]
            st.success(f"Culture recommandée : **{best['crop']}** ({best['predicted_yield_t_ha']} t/ha estimé)")

            st.bar_chart(
                recos.set_index("crop")["predicted_yield_t_ha"],
                color="#3b7a57",
            )

            st.dataframe(
                recos.rename(
                    columns={
                        "rank": "Rang",
                        "crop": "Culture",
                        "predicted_yield_t_ha": "Rendement estimé (t/ha)",
                    }
                ).set_index("Rang"),
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
