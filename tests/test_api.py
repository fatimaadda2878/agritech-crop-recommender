"""
Tests unitaires de l'API FastAPI (endpoints /health, /metadata, /predict,
/recommend). Ces tests sont exécutés automatiquement par le pipeline CI/CD
à chaque push (voir .github/workflows/ci-cd.yml).
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "api"))

from main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # Le contexte `with` déclenche le lifespan de l'API (chargement du
    # modèle et des métadonnées), comme le ferait un vrai déploiement.
    with TestClient(app) as c:
        yield c


@pytest.fixture
def valid_predict_payload():
    return {
        "region": "North",
        "soil_type": "Loam",
        "rainfall_mm": 600.0,
        "temperature_celsius": 25.0,
        "fertilizer_used": True,
        "irrigation_used": True,
        "weather_condition": "Sunny",
        "days_to_harvest": 110,
        "crop": "Wheat",
    }


@pytest.fixture
def valid_recommend_payload():
    return {
        "region": "North",
        "soil_type": "Loam",
        "rainfall_mm": 600.0,
        "temperature_celsius": 25.0,
        "fertilizer_used": True,
        "irrigation_used": True,
        "weather_condition": "Sunny",
        "days_to_harvest": 110,
    }


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_metadata(client):
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "Wheat" in data["valid_crops"]
    assert set(data["valid_regions"]) == {"West", "South", "North", "East"}
    assert "rmse" in data["model_metrics"]


def test_predict_returns_positive_yield(client, valid_predict_payload):
    response = client.post("/predict", json=valid_predict_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["crop"] == "Wheat"
    assert data["predicted_yield_t_ha"] >= 0
    assert isinstance(data["predicted_yield_t_ha"], float)


def test_predict_invalid_crop(client, valid_predict_payload):
    valid_predict_payload["crop"] = "Truffle"
    response = client.post("/predict", json=valid_predict_payload)
    assert response.status_code == 422


def test_predict_invalid_region(client, valid_predict_payload):
    valid_predict_payload["region"] = "Atlantis"
    response = client.post("/predict", json=valid_predict_payload)
    assert response.status_code == 422


def test_predict_missing_field(client, valid_predict_payload):
    del valid_predict_payload["rainfall_mm"]
    response = client.post("/predict", json=valid_predict_payload)
    assert response.status_code == 422


def test_recommend_returns_all_crops_ranked(client, valid_recommend_payload):
    response = client.post("/recommend", json=valid_recommend_payload)
    assert response.status_code == 200
    data = response.json()
    recos = data["recommendations"]
    assert len(recos) == 6  # 6 cultures possibles
    # Vérifie que le classement est bien décroissant par rendement
    yields = [r["predicted_yield_t_ha"] for r in recos]
    assert yields == sorted(yields, reverse=True)
    # Vérifie que les rangs sont bien 1..6 dans l'ordre
    ranks = [r["rank"] for r in recos]
    assert ranks == list(range(1, 7))


def test_recommend_invalid_soil_type(client, valid_recommend_payload):
    valid_recommend_payload["soil_type"] = "Diamond"
    response = client.post("/recommend", json=valid_recommend_payload)
    assert response.status_code == 422


@pytest.mark.parametrize("crop", ["Barley", "Cotton", "Maize", "Rice", "Soybean", "Wheat"])
def test_predict_all_crops(client, valid_predict_payload, crop):
    """Chaque culture (y compris celles sans correspondance FAO directe,
    Barley et Cotton) doit produire une prédiction valide."""
    valid_predict_payload["crop"] = crop
    response = client.post("/predict", json=valid_predict_payload)
    assert response.status_code == 200
    assert response.json()["predicted_yield_t_ha"] >= 0
