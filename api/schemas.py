"""Schémas Pydantic pour l'API de prédiction/recommandation de rendement."""
from pydantic import BaseModel, Field


class ParcelContext(BaseModel):
    """Conditions d'une parcelle, communes aux deux endpoints."""

    region: str = Field(..., examples=["North"], description="Région de la parcelle")
    soil_type: str = Field(..., examples=["Loam"], description="Type de sol")
    rainfall_mm: float = Field(..., ge=0, le=2000, examples=[600.0], description="Pluviométrie (mm)")
    temperature_celsius: float = Field(..., ge=-10, le=55, examples=[25.0], description="Température moyenne (°C)")
    fertilizer_used: bool = Field(..., examples=[True], description="Engrais utilisé sur la parcelle")
    irrigation_used: bool = Field(..., examples=[True], description="Irrigation utilisée sur la parcelle")
    weather_condition: str = Field(..., examples=["Sunny"], description="Condition météo dominante")
    days_to_harvest: int = Field(..., ge=1, le=365, examples=[110], description="Nombre de jours avant récolte")


class PredictRequest(ParcelContext):
    """Requête de prédiction : conditions de parcelle + culture choisie."""

    crop: str = Field(..., examples=["Wheat"], description="Culture sélectionnée")


class PredictResponse(BaseModel):
    crop: str
    predicted_yield_t_ha: float
    model_type: str


class RecommendRequest(ParcelContext):
    """Requête de recommandation : conditions de parcelle uniquement."""
    pass


class CropRecommendation(BaseModel):
    crop: str
    predicted_yield_t_ha: float
    rank: int


class RecommendResponse(BaseModel):
    recommendations: list[CropRecommendation]
    model_type: str


class Metadata(BaseModel):
    valid_crops: list[str]
    valid_regions: list[str]
    valid_soil_types: list[str]
    valid_weather: list[str]
    model_type: str
    model_metrics: dict
