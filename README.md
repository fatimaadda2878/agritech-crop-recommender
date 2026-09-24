# 🌾 Agritech Crop Recommender

Application d'aide à la décision pour les agriculteurs, développée pour **Agritech Answers**. Elle combine :

- une **fonction de prédiction** : rendement estimé (t/ha) pour une culture choisie, selon les conditions d'une parcelle ;
- une **fonction de recommandation** : classement des 6 cultures possibles (blé, orge, coton, maïs, riz, soja) par rendement estimé, pour les conditions d'une parcelle donnée.

[![CI/CD - Agritech Crop Recommender](https://github.com/fatimaadda2878/agritech-crop-recommender/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/fatimaadda2878/agritech-crop-recommender/actions/workflows/ci-cd.yml)

## 🚀 Démo en ligne

- **Application (Streamlit Community Cloud) : [https://agritech-crop-recommender-api.streamlit.app/](https://agritech-crop-recommender-api.streamlit.app/)**
  Interface agriculteur : renseigner les conditions de la parcelle dans le panneau de gauche, puis choisir *Prédiction* (rendement d'une culture) ou *Recommandation* (classement des 6 cultures).
- **API (Render) : [https://agritech-crop-api.onrender.com/docs](https://agritech-crop-api.onrender.com/docs)**
  Documentation interactive (Swagger) : ouvrir `/predict` ou `/recommend`, cliquer sur *Try it out*, modifier les valeurs d'exemple puis *Execute*.

L'application Streamlit interroge l'API Render à chaque prédiction.

> ℹ️ Hébergement gratuit : les services se mettent en veille après une période d'inactivité. Le premier appel peut donc prendre environ une minute, le temps qu'ils redémarrent.

## Architecture

```
                    ┌──────────────────────┐
                    │   Données brutes     │
                    │  (crop_yield.csv +   │
                    │   yield_df.csv)      │
                    └──────────┬───────────┘
                               │ src/data_prep.py
                               ▼
                    ┌──────────────────────┐
                    │  Dataset consolidé   │
                    │ data/processed/*.csv │
                    └──────────┬───────────┘
                               │ src/train.py (+ MLflow)
                               ▼
                    ┌──────────────────────┐
                    │  Modèle entraîné     │
                    │  models/model.pkl    │
                    └──────────┬───────────┘
                               │ copié dans
                               ▼
        ┌───────────────────────────────────────┐
        │        api/  (FastAPI, Docker)         │
        │   POST /predict     POST /recommend    │
        └──────────────────┬──────────────────────┘
                            │ requêtes HTTP
                            ▼
        ┌───────────────────────────────────────┐
        │        app/ (Streamlit)                │
        │   Interface agriculteur                │
        └───────────────────────────────────────┘
```

## Structure du dépôt

```
├── data/
│   ├── raw/                  # Données brutes (non versionnées, voir "Récupérer les données")
│   └── processed/            # Dataset consolidé (généré, non versionné)
├── notebooks/
│   └── 01_data_merging_eda.ipynb   # Exploration, fusion des 2 datasets, ACP
├── src/
│   ├── data_prep.py           # Logique de fusion/nettoyage (partagée notebook + training)
│   └── train.py                # Entraînement, comparaison, optimisation, suivi MLflow
├── api/
│   ├── main.py                 # API FastAPI (/predict, /recommend, /metadata, /health)
│   ├── schemas.py               # Schémas Pydantic
│   ├── models/                  # Modèle + métadonnées servis par l'API
│   ├── Dockerfile
│   └── requirements.txt
├── app/
│   ├── app.py                   # Application Streamlit (front-end)
│   └── requirements.txt
├── tests/
│   └── test_api.py              # Tests unitaires pytest de l'API
├── .github/workflows/ci-cd.yml  # Pipeline CI/CD GitHub Actions
├── docs/CI_CD.md                # Documentation détaillée du pipeline
├── reports/                     # Figures, comparatif de modèles, rapport métier PDF
├── mlflow_screenshots/          # Captures d'écran MLflow (preuves d'expérimentation)
└── requirements.txt              # Dépendances du pipeline data/ML
```

## Récupérer les données

Les fichiers de données bruts ne sont pas versionnés dans Git (volume trop important). Pour reconstituer `data/raw/` :

1. `Agriculture CropYield Dataset` → placer `crop_yield.csv` dans `data/raw/`.
2. `CropYield Prediction Dataset` → placer `yield_df.csv` dans `data/raw/`.

Puis reconstruire le dataset consolidé et le modèle :

```bash
pip install -r requirements.txt
python src/data_prep.py     # écrit data/processed/merged_dataset.csv
python src/train.py         # entraîne le modèle, écrit models/ et suit les runs dans MLflow
```

## Lancer le projet en local

### 1. L'API

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Une fois l'API lancée sur votre machine, sa documentation interactive est accessible à l'adresse `http://localhost:8000/docs` (uniquement pendant que l'API tourne en local). Pour tester sans rien installer, utiliser la [version en ligne](https://agritech-crop-api.onrender.com/docs).

### 2. L'application Streamlit

```bash
cd app
pip install -r requirements.txt
API_URL=http://localhost:8000 streamlit run app.py
```

### 3. Avec Docker (API uniquement)

```bash
cd api
docker build -t agritech-crop-api .
docker run -p 8000:8000 agritech-crop-api
```

### 4. Déploiement en ligne (gratuit)

Docker Hub sert seulement à stocker une image, pas à l'exécuter publiquement :
c'est pour ça qu'un déploiement réel utilise une vraie plateforme d'hébergement.
Ici, l'API et l'interface sont déployées gratuitement sur deux plateformes,
chacune connectée directement à ce dépôt GitHub et redéployée automatiquement
à chaque push sur `main` :

**API (FastAPI) sur [Render](https://render.com) :**
1. Créer un compte gratuit sur render.com (connexion avec GitHub conseillée).
2. *New +* → *Web Service* → sélectionner ce dépôt.
3. *Environment* : `Docker`, *Root Directory* : `api`, plan `Free`.
4. Render construit `api/Dockerfile` et fournit une URL publique (ex.
   `https://agritech-crop-api.onrender.com`). Elle se redéploie automatiquement
   à chaque push sur `main`. Le service gratuit se met en veille après 15 min
   d'inactivité et se réveille en ~1 minute au premier appel suivant.

**Interface (Streamlit) sur [Streamlit Community Cloud](https://share.streamlit.io) :**
1. Se connecter avec son compte GitHub sur share.streamlit.io.
2. *Create app* → sélectionner ce dépôt, branche `main`, fichier principal
   `app/app.py`.
3. Dans *Advanced settings* → *Secrets*, ajouter :
   `API_URL = "https://agritech-crop-api.onrender.com"`
4. Déployer. L'application est accessible sur
   [https://agritech-crop-recommender-api.streamlit.app/](https://agritech-crop-recommender-api.streamlit.app/).

L'interface peut aussi être lancée en local tout en interrogeant l'API en ligne :

```bash
cd app
pip install -r requirements.txt
API_URL=https://agritech-crop-api.onrender.com streamlit run app.py
```

Le job optionnel `deploy` du pipeline CI/CD (publication Docker Hub) reste
désactivé par défaut ; voir [`docs/CI_CD.md`](docs/CI_CD.md) pour l'activer si besoin.

### 5. Visualiser les expérimentations MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

## Tests

```bash
pip install -r api/requirements.txt pytest httpx
pytest tests/ -v
```

## Notebooks et rapport

- `notebooks/01_data_merging_eda.ipynb` : exploration des 2 sources, stratégie de fusion justifiée, ACP complète (cercle des corrélations, scree plot, projection des individus).
- `reports/rapport_metier.pdf` : rapport de synthèse pour public non technique (résultats, variables clés, recommandations agronomiques, captures MLflow).
- `reports/model_comparison.csv` : comparatif chiffré des modèles testés.

## CI/CD

Voir [`docs/CI_CD.md`](docs/CI_CD.md) pour le détail du pipeline (tests → build Docker → publication Docker Hub optionnelle) et de la procédure de déploiement en ligne (API sur Render, interface Streamlit).

## Choix méthodologiques principaux

- **Fusion des données** : les deux jeux de données ne partagent pas de clé pays/année (absente du dataset parcelle) ; la fusion se fait donc par culture, en enrichissant chaque parcelle avec des benchmarks climatiques/agronomiques mondiaux calculés à partir des données FAO.
- **Échantillonnage d'entraînement** : 200 000 parcelles (sur 1 000 000) pour l'entraînement/optimisation, un choix documenté dans `src/train.py` et justifié par la stabilité des métriques obtenues.
- **Modèle retenu** : régression Ridge optimisée (RMSE ≈ 0,50 t/ha, R² ≈ 0,915), choisie pour sa simplicité et son interprétabilité à performance égale avec des modèles plus complexes (Random Forest, Gradient Boosting).
