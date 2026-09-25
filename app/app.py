"""
Application Streamlit — Agritech Answers.

Interface simple et visuelle permettant à un agriculteur de :
  - Mode "Prédiction" : estimer le rendement d'une culture qu'il a déjà choisie.
  - Mode "Recommandation" : obtenir un classement indicatif des cultures selon
    leur rendement estimé pour les conditions de sa parcelle. Le classement
    tient compte de l'erreur du modèle : quand les écarts entre cultures sont
    plus faibles que cette erreur, les cultures sont présentées comme
    équivalentes. Il porte sur le rendement, pas sur la rentabilité (prix de
    vente et coûts de production ne sont pas connus du modèle).

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

    # Erreur moyenne du modèle, utilisée pour nuancer les résultats affichés
    model_rmse = float(metadata["model_metrics"]["rmse"])

    mode = st.radio(
        "Que souhaitez-vous faire ?",
        options=["🔮 Prédiction", "📊 Recommandation"],
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
                value=f"{result['predicted_yield_t_ha']:.2f} t/ha",
            )
            st.caption(
                f"Erreur moyenne du modèle (RMSE) : environ ± {model_rmse:.2f} t/ha. "
                "Le rendement réel peut donc s'écarter de cette estimation."
            )

    else:
        st.subheader("Comparer les cultures pour votre parcelle")
        st.write(
            "L'application estime le rendement de **chaque culture possible** avec les "
            "conditions de parcelle renseignées à gauche, puis les classe par rendement "
            "estimé."
        )
        st.info(
            "Ce classement est **indicatif**. Il porte uniquement sur le **rendement "
            "(t/ha)**, pas sur la rentabilité : les prix de vente et les coûts de "
            "production ne sont pas pris en compte. De plus, le modèle a une erreur "
            f"moyenne d'environ ± {model_rmse:.2f} t/ha : deux cultures dont les "
            "rendements estimés sont plus proches que cette erreur ne peuvent pas être "
            "départagées."
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
            spread = recos["predicted_yield_t_ha"].max() - recos["predicted_yield_t_ha"].min()
            # Écart de chaque culture avec la première du classement
            recos["gap_to_first"] = best["predicted_yield_t_ha"] - recos["predicted_yield_t_ha"]
            # Cultures qu'on ne peut pas distinguer de la première compte tenu de
            # l'erreur du modèle (écart inférieur au RMSE)
            equivalent = recos[recos["gap_to_first"] < model_rmse]

            if spread < model_rmse:
                st.warning(
                    f"**Pas de différence nette entre les cultures.** Les rendements estimés "
                    f"vont de {recos['predicted_yield_t_ha'].min():.2f} à "
                    f"{recos['predicted_yield_t_ha'].max():.2f} t/ha, soit un écart de "
                    f"{spread:.2f} t/ha, plus faible que l'erreur du modèle "
                    f"(± {model_rmse:.2f} t/ha). Dans ces conditions, les données ne "
                    "permettent pas de dire qu'une culture donnera un meilleur rendement "
                    "qu'une autre : le choix peut se faire sur d'autres critères (prix, "
                    "coûts, débouchés, rotation des cultures)."
                )
            elif len(equivalent) > 1:
                others = ", ".join(equivalent["crop"].iloc[1:])
                st.info(
                    f"**{best['crop']}** a le rendement estimé le plus élevé "
                    f"({best['predicted_yield_t_ha']:.2f} t/ha), mais son avance sur "
                    f"{others} est plus faible que l'erreur du modèle "
                    f"(± {model_rmse:.2f} t/ha) : ces cultures sont à considérer comme "
                    "équivalentes en rendement."
                )
            else:
                second = recos.iloc[1]
                st.success(
                    f"**{best['crop']}** a le rendement estimé le plus élevé "
                    f"({best['predicted_yield_t_ha']:.2f} t/ha), avec une avance de "
                    f"{best['predicted_yield_t_ha'] - second['predicted_yield_t_ha']:.2f} t/ha "
                    f"sur {second['crop']}, supérieure à l'erreur du modèle "
                    f"(± {model_rmse:.2f} t/ha). Ce résultat reste indicatif et ne tient "
                    "pas compte de la rentabilité."
                )

            st.bar_chart(
                recos.set_index("crop")["predicted_yield_t_ha"],
                color="#3b7a57",
            )

            table = recos.assign(
                distinct=recos["gap_to_first"].map(
                    lambda gap: "Oui" if gap >= model_rmse else "Non"
                )
            )
            table.loc[table.index[0], "distinct"] = "—"
            st.dataframe(
                table.rename(
                    columns={
                        "rank": "Rang",
                        "crop": "Culture",
                        "predicted_yield_t_ha": "Rendement estimé (t/ha)",
                        "gap_to_first": "Écart avec la 1re (t/ha)",
                        "distinct": "Écart > erreur du modèle ?",
                    }
                ).set_index("Rang"),
                column_config={
                    "Rendement estimé (t/ha)": st.column_config.NumberColumn(format="%.2f"),
                    "Écart avec la 1re (t/ha)": st.column_config.NumberColumn(format="%.2f"),
                },
                width="stretch",
            )
            st.caption(
                "Lecture : « Non » signifie que l'écart avec la culture classée première "
                "est plus faible que l'erreur du modèle, les deux cultures ne peuvent donc "
                "pas être départagées."
            )

if __name__ == "__main__":
    main()
