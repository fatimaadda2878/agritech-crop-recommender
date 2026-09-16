"""
API FastAPI — Agritech Answers, moteur de prédiction & recommandation de
rendement agricole.

Endpoints :
    GET  /health     -> vérification de l'état de l'API
    GET  /metadata    -> valeurs valides (cultures, régions, ...) + infos modèle
    POST /predict     -> rendement estimé pour UNE culture donnée
    POST /recommend   -> classement de TOUTES les cultures par rendement estimé

Le modèle (pipeline scikit-learn complet : préprocessing + régression) est
chargé une seule fois au démarrage depuis models/model.pkl. Cette API ne
contient aucune logique métier "cachée" côté front-end : c'est le seul
point d'entrée pour obtenir une prédiction.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    Metadata,
    PredictRequest,
    PredictResponse,
    RecommendRequest,
    RecommendResponse,
    CropRecommendation,
)

MODELS_DIR = Path(__file__).parent / "models"

# --- Chargement des artefacts au démarrage ---
_model = None
_metadata = None
_benchmarks = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _metadata, _benchmarks
    _model = joblib.load(MODELS_DIR / "model.pkl")
    with open(MODELS_DIR / "model_metadata.json") as f:
        _metadata = json.load(f)
    _benchmarks = pd.read_csv(MODELS_DIR / "crop_benchmarks.csv")
    yield


app = FastAPI(
    title="Agritech Answers — Crop Yield API",
    description="API de prédiction de rendement et de recommandation de culture.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _crop_norm(crop: str) -> str:
    return crop.strip().lower()


def _validate_categorical(value: str, valid_values: list[str], field_name: str):
    if value not in valid_values:
        raise HTTPException(
            status_code=422,
            detail=f"Valeur invalide pour '{field_name}': '{value}'. Valeurs possibles: {valid_values}",
        )


def _build_feature_row(context: dict, crop: str) -> pd.DataFrame:
    """Construit une ligne de features identique à celle utilisée à
    l'entraînement, à partir du contexte parcelle + de la culture testée."""
    crop_norm = _crop_norm(crop)
    bench_row = _benchmarks[_benchmarks["Item_norm"] == crop_norm]
    if bench_row.empty:
        raise HTTPException(status_code=422, detail=f"Culture inconnue: {crop}")
    bench = bench_row.iloc[0]

    rainfall_gap_pct = (context["rainfall_mm"] - bench["world_avg_rainfall_mm"]) / bench["world_avg_rainfall_mm"] * 100
    temperature_gap_c = context["temperature_celsius"] - bench["world_avg_temp_c"]

    row = {
        "Rainfall_mm": context["rainfall_mm"],
        "Temperature_Celsius": context["temperature_celsius"],
        "Days_to_Harvest": context["days_to_harvest"],
        "world_avg_rainfall_mm": bench["world_avg_rainfall_mm"],
        "world_avg_temp_c": bench["world_avg_temp_c"],
        "world_avg_pesticides_tonnes": bench["world_avg_pesticides_tonnes"],
        "world_avg_yield_t_ha": bench["world_avg_yield_t_ha"],
        "rainfall_gap_pct": rainfall_gap_pct,
        "Fertilizer_Used": context["fertilizer_used"],
        "Irrigation_Used": context["irrigation_used"],
        "Region": context["region"],
        "Soil_Type": context["soil_type"],
        "Weather_Condition": context["weather_condition"],
        "Crop": crop,
    }
    return pd.DataFrame([row])[_metadata["feature_columns"]]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/metadata", response_model=Metadata)
def metadata():
    return Metadata(
        valid_crops=_metadata["valid_crops"],
        valid_regions=_metadata["valid_regions"],
        valid_soil_types=_metadata["valid_soil_types"],
        valid_weather=_metadata["valid_weather"],
        model_type=_metadata["best_model_type"],
        model_metrics=_metadata["metrics"],
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    _validate_categorical(request.crop, _metadata["valid_crops"], "crop")
    _validate_categorical(request.region, _metadata["valid_regions"], "region")
    _validate_categorical(request.soil_type, _metadata["valid_soil_types"], "soil_type")
    _validate_categorical(request.weather_condition, _metadata["valid_weather"], "weather_condition")

    context = request.model_dump()
    X = _build_feature_row(context, request.crop)
    prediction = float(_model.predict(X)[0])
    prediction = max(prediction, 0.0)

    return PredictResponse(
        crop=request.crop,
        predicted_yield_t_ha=round(prediction, 3),
        model_type=_metadata["best_model_type"],
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    _validate_categorical(request.region, _metadata["valid_regions"], "region")
    _validate_categorical(request.soil_type, _metadata["valid_soil_types"], "soil_type")
    _validate_categorical(request.weather_condition, _metadata["valid_weather"], "weather_condition")

    context = request.model_dump()
    rows = []
    for crop in _metadata["valid_crops"]:
        X = _build_feature_row(context, crop)
        prediction = max(float(_model.predict(X)[0]), 0.0)
        rows.append((crop, prediction))

    rows.sort(key=lambda r: r[1], reverse=True)
    recommendations = [
        CropRecommendation(crop=crop, predicted_yield_t_ha=round(pred, 3), rank=i + 1)
        for i, (crop, pred) in enumerate(rows)
    ]

    return RecommendResponse(recommendations=recommendations, model_type=_metadata["best_model_type"])
