"""
Module de préparation des données pour le projet Agritech Crop Recommender.

Ce module contient toute la logique de fusion des deux jeux de données
(Agriculture CropYield Dataset + CropYield Prediction Dataset). Il est
importé à la fois par le notebook d'exploration (notebooks/01_...) et par
le script d'entraînement (src/train.py), pour garantir que le même
pipeline de préparation est utilisé partout (reproductibilité MLOps).
"""
from __future__ import annotations

import pandas as pd
import numpy as np

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

# Correspondance des noms de cultures entre les deux jeux de données.
# Le CropYield Prediction Dataset (FAO) utilise des noms différents pour
# certaines cultures pourtant identiques (ex: "Rice, paddy" == "Rice").
CROP_NAME_MAPPING = {
    "rice, paddy": "rice",
    "soybeans": "soybean",
    "maize": "maize",
    "wheat": "wheat",
}

# Cultures du dataset principal (parcelle) qui n'ont pas d'équivalent
# direct dans les données FAO (Barley, Cotton). Pour ces cultures, on
# utilisera la moyenne mondiale tous produits confondus comme valeur de
# repli (limitation documentée dans le rapport et le notebook).
CROPS_WITHOUT_FAO_MATCH = {"barley", "cotton"}


def load_raw_datasets(raw_dir: str = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les deux jeux de données bruts."""
    df_parcels = pd.read_csv(f"{raw_dir}/crop_yield.csv")
    df_world = pd.read_csv(f"{raw_dir}/yield_df.csv", index_col=0)
    return df_parcels, df_world


def clean_parcels_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le dataset 'Agriculture CropYield' (niveau parcelle)."""
    df = df.copy()
    df = df.drop_duplicates()

    # Le rendement ne peut pas être négatif physiquement. Quelques valeurs
    # négatives (bruit de simulation) sont présentes dans le jeu de
    # données brut : on les ramène à 0 plutôt que de supprimer la ligne
    # (les autres variables restent informatives).
    n_negative = (df["Yield_tons_per_hectare"] < 0).sum()
    df["Yield_tons_per_hectare"] = df["Yield_tons_per_hectare"].clip(lower=0)

    df["Crop_norm"] = df["Crop"].str.strip().str.lower()
    df.attrs["n_negative_yield_clipped"] = int(n_negative)
    return df


def clean_world_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le dataset 'CropYield Prediction' (niveau pays/année)."""
    df = df.copy()
    n_dup = df.duplicated().sum()
    df = df.drop_duplicates()
    df["Item_norm"] = df["Item"].str.strip().str.lower().replace(CROP_NAME_MAPPING)
    df.attrs["n_duplicates_removed"] = int(n_dup)
    return df


def build_crop_benchmarks(df_world_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Construit une table de référence "monde" par culture, agrégée à
    partir du CropYield Prediction Dataset (moyenne tous pays et toutes
    années confondus, 1990-2013).

    Ces indicateurs servent de PROXY pour enrichir le dataset parcelle,
    qui ne contient ni pays ni année (donc pas de clé de jointure directe
    pays/année possible) : pour chaque culture, on calcule la pluviométrie
    moyenne mondiale, la température moyenne mondiale, l'usage moyen de
    pesticides (au niveau national, tous produits confondus) et le
    rendement moyen mondial.
    """
    agg = (
        df_world_clean.groupby("Item_norm")
        .agg(
            world_avg_rainfall_mm=("average_rain_fall_mm_per_year", "mean"),
            world_avg_temp_c=("avg_temp", "mean"),
            world_avg_pesticides_tonnes=("pesticides_tonnes", "mean"),
            world_avg_yield_hg_ha=("hg/ha_yield", "mean"),
        )
        .reset_index()
    )
    agg["world_avg_yield_t_ha"] = agg["world_avg_yield_hg_ha"] / 10_000
    agg = agg.drop(columns=["world_avg_yield_hg_ha"])

    # Valeur de repli (moyenne globale tous produits) pour les cultures
    # sans correspondance FAO (Barley, Cotton).
    fallback = agg[
        ["world_avg_rainfall_mm", "world_avg_temp_c", "world_avg_pesticides_tonnes", "world_avg_yield_t_ha"]
    ].mean()

    rows = []
    for crop in CROPS_WITHOUT_FAO_MATCH:
        rows.append({"Item_norm": crop, **fallback.to_dict()})
    fallback_df = pd.DataFrame(rows)

    benchmarks = pd.concat([agg, fallback_df], ignore_index=True)
    return benchmarks


def merge_datasets(df_parcels_clean: pd.DataFrame, benchmarks: pd.DataFrame) -> pd.DataFrame:
    """
    Fusionne le dataset parcelle avec les benchmarks mondiaux par culture,
    puis calcule les variables d'écart (proxy) utilisées comme features
    supplémentaires par le modèle.
    """
    merged = df_parcels_clean.merge(
        benchmarks, left_on="Crop_norm", right_on="Item_norm", how="left"
    ).drop(columns=["Item_norm"])

    # Variables d'écart par rapport aux références mondiales de la culture :
    # elles indiquent si la parcelle est plus/moins arrosée, plus chaude/
    # froide que la moyenne mondiale pour cette culture.
    merged["rainfall_gap_pct"] = (
        (merged["Rainfall_mm"] - merged["world_avg_rainfall_mm"]) / merged["world_avg_rainfall_mm"]
    ) * 100
    merged["temperature_gap_c"] = merged["Temperature_Celsius"] - merged["world_avg_temp_c"]

    return merged


def build_merged_dataset(raw_dir: str = RAW_DIR) -> tuple[pd.DataFrame, dict]:
    """Pipeline complet : charge, nettoie, fusionne. Retourne aussi des
    statistiques utiles pour documenter les choix (notebook / rapport)."""
    df_parcels, df_world = load_raw_datasets(raw_dir)

    df_parcels_clean = clean_parcels_dataset(df_parcels)
    df_world_clean = clean_world_dataset(df_world)

    benchmarks = build_crop_benchmarks(df_world_clean)
    merged = merge_datasets(df_parcels_clean, benchmarks)

    stats = {
        "n_rows_parcels_raw": len(df_parcels),
        "n_rows_parcels_clean": len(df_parcels_clean),
        "n_negative_yield_clipped": df_parcels_clean.attrs.get("n_negative_yield_clipped", 0),
        "n_duplicates_removed_world": df_world_clean.attrs.get("n_duplicates_removed", 0),
        "n_rows_merged": len(merged),
        "n_missing_after_merge": int(merged.isna().sum().sum()),
        "crops_with_fao_match": sorted(set(df_parcels_clean["Crop_norm"].unique()) - CROPS_WITHOUT_FAO_MATCH),
        "crops_fallback": sorted(CROPS_WITHOUT_FAO_MATCH),
    }
    return merged, stats


FEATURE_COLUMNS_NUMERIC = [
    "Rainfall_mm",
    "Temperature_Celsius",
    "Days_to_Harvest",
    "world_avg_rainfall_mm",
    "world_avg_temp_c",
    "world_avg_pesticides_tonnes",
    "world_avg_yield_t_ha",
    "rainfall_gap_pct",
    "temperature_gap_c",
]
FEATURE_COLUMNS_BOOL = ["Fertilizer_Used", "Irrigation_Used"]
FEATURE_COLUMNS_CATEGORICAL = ["Region", "Soil_Type", "Weather_Condition", "Crop"]
TARGET_COLUMN = "Yield_tons_per_hectare"

ALL_CROPS = ["Barley", "Cotton", "Maize", "Rice", "Soybean", "Wheat"]
ALL_REGIONS = ["West", "South", "North", "East"]
ALL_SOIL_TYPES = ["Sandy", "Clay", "Loam", "Silt", "Peaty", "Chalky"]
ALL_WEATHER = ["Cloudy", "Rainy", "Sunny"]


if __name__ == "__main__":
    merged, stats = build_merged_dataset()
    print(stats)
    import os

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = f"{PROCESSED_DIR}/merged_dataset.csv"
    merged.to_csv(out_path, index=False)
    print(f"Dataset consolidé écrit dans {out_path} ({len(merged)} lignes)")
