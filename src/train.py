"""
Entraînement, comparaison et optimisation des modèles de prédiction du
rendement agricole, avec suivi complet des expérimentations dans MLflow.

Usage:
    python src/train.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_prep import (
    ALL_CROPS,
    ALL_REGIONS,
    ALL_SOIL_TYPES,
    ALL_WEATHER,
    FEATURE_COLUMNS_BOOL,
    FEATURE_COLUMNS_CATEGORICAL,
    FEATURE_COLUMNS_NUMERIC,
    TARGET_COLUMN,
    build_crop_benchmarks,
    build_merged_dataset,
    clean_world_dataset,
    load_raw_datasets,
)

SEED = 42
np.random.seed(SEED)

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
DATA_PROCESSED = Path("data/processed")
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# On travaille sur un échantillon représentatif du dataset consolidé
# (1 000 000 lignes) plutôt que sur la totalité : l'environnement
# d'entraînement dispose de ressources limitées (2 coeurs CPU), et les
# données étant très homogènes (peu de bruit, aucune valeur manquante),
# un échantillon de 200 000 lignes stratifié par culture est largement
# suffisant pour obtenir des métriques stables. Ce choix est documenté
# ici et dans le rapport métier.
SAMPLE_SIZE = 200_000

MLFLOW_EXPERIMENT = "crop_yield_prediction"


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, FEATURE_COLUMNS_NUMERIC[:-1]),  # sans la target
            ("bool", "passthrough", FEATURE_COLUMNS_BOOL),
            ("cat", categorical_transformer, FEATURE_COLUMNS_CATEGORICAL),
        ]
    )
    return preprocessor


def load_training_data() -> pd.DataFrame:
    csv_path = DATA_PROCESSED / "merged_dataset.csv"
    if csv_path.exists():
        print(f"Chargement du dataset consolidé existant : {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print("Dataset consolidé introuvable, reconstruction depuis les données brutes...")
        df, _ = build_merged_dataset()
        df.to_csv(csv_path, index=False)
    return df


def sample_data(df: pd.DataFrame, n: int) -> pd.DataFrame:
    n = min(n, len(df))
    frac = n / len(df)
    parts = [group.sample(frac=frac, random_state=SEED) for _, group in df.groupby("Crop")]
    return pd.concat(parts, ignore_index=True)


def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def log_dataset_context(mlflow_active_run, n_total, n_sample):
    mlflow.log_param("dataset_total_rows", n_total)
    mlflow.log_param("dataset_sample_rows", n_sample)
    mlflow.log_param("random_seed", SEED)


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df = load_training_data()
    n_total = len(df)
    df_sample = sample_data(df, SAMPLE_SIZE)
    print(f"Dataset total: {n_total} lignes | échantillon d'entraînement: {len(df_sample)} lignes")

    feature_cols = FEATURE_COLUMNS_NUMERIC[:-1] + FEATURE_COLUMNS_BOOL + FEATURE_COLUMNS_CATEGORICAL
    X = df_sample[feature_cols]
    y = df_sample[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=df_sample["Crop"]
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    candidates = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=100, max_depth=12, random_state=SEED, n_jobs=-1
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=200, max_depth=8, random_state=SEED
        ),
    }

    results = []
    for name, model in candidates.items():
        with mlflow.start_run(run_name=f"baseline_{name}") as run:
            t0 = time.time()
            pipeline = Pipeline([("preprocessor", build_preprocessor()), ("model", model)])
            pipeline.fit(X_train, y_train)
            train_time = time.time() - t0

            y_pred = pipeline.predict(X_test)
            metrics = evaluate(y_test, y_pred)

            mlflow.log_param("model_type", name)
            mlflow.log_params({f"hp_{k}": v for k, v in model.get_params().items() if not callable(v)})
            log_dataset_context(run, n_total, len(df_sample))
            mlflow.log_metric("rmse", metrics["rmse"])
            mlflow.log_metric("mae", metrics["mae"])
            mlflow.log_metric("r2", metrics["r2"])
            mlflow.log_metric("train_time_seconds", train_time)
            mlflow.set_tag("stage", "baseline")
            mlflow.set_tag("mlflow.note.content", f"Modèle de base {name}, sans optimisation d'hyperparamètres.")

            print(f"[baseline] {name}: RMSE={metrics['rmse']:.4f} R2={metrics['r2']:.4f} ({train_time:.1f}s)")
            results.append({"name": name, **metrics, "run_id": run.info.run_id})

    results_df = pd.DataFrame(results).sort_values("rmse")
    print("\nClassement des modèles de base (par RMSE croissant) :")
    print(results_df)

    best_baseline_name = results_df.iloc[0]["name"]
    print(f"\nMeilleur modèle de base : {best_baseline_name} -> optimisation des hyperparamètres")

    if best_baseline_name == "random_forest":
        base_model = RandomForestRegressor(random_state=SEED, n_jobs=-1)
        param_distributions = {
            "model__n_estimators": [100, 150, 200],
            "model__max_depth": [8, 12, 16, None],
            "model__min_samples_leaf": [1, 2, 5],
        }
    elif best_baseline_name == "hist_gradient_boosting":
        base_model = HistGradientBoostingRegressor(random_state=SEED)
        param_distributions = {
            "model__max_iter": [150, 200, 300],
            "model__max_depth": [6, 8, 10, None],
            "model__learning_rate": [0.03, 0.06, 0.1],
            "model__l2_regularization": [0.0, 0.1, 0.5],
        }
    else:
        # La régression linéaire simple n'a pas d'hyperparamètre à optimiser :
        # on optimise plutôt sa variante régularisée (Ridge), ce qui permet
        # quand même de démontrer la démarche de recherche d'hyperparamètres
        # tout en restant cohérent avec le fait que la relation semble
        # majoritairement linéaire sur ce jeu de données.
        best_baseline_name = "ridge_regression"
        base_model = Ridge(random_state=SEED)
        param_distributions = {
            "model__alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
            "model__solver": ["auto", "svd", "cholesky"],
        }

    tuning_pipeline = Pipeline([("preprocessor", build_preprocessor()), ("model", base_model)])

    search = RandomizedSearchCV(
        tuning_pipeline,
        param_distributions=param_distributions,
        n_iter=8,
        scoring="neg_root_mean_squared_error",
        cv=3,
        random_state=SEED,
        n_jobs=1,
        verbose=1,
    )

    with mlflow.start_run(run_name=f"tuning_{best_baseline_name}") as parent_run:
        t0 = time.time()
        search.fit(X_train, y_train)
        tuning_time = time.time() - t0

        for i, (params, mean_score) in enumerate(
            zip(search.cv_results_["params"], search.cv_results_["mean_test_score"])
        ):
            with mlflow.start_run(run_name=f"candidate_{i}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("cv_neg_rmse", float(mean_score))
                mlflow.set_tag("stage", "hyperparameter_search")

        best_pipeline = search.best_estimator_
        y_pred = best_pipeline.predict(X_test)
        metrics = evaluate(y_test, y_pred)

        mlflow.log_param("model_type", best_baseline_name)
        mlflow.log_params(search.best_params_)
        log_dataset_context(parent_run, n_total, len(df_sample))
        mlflow.log_metric("rmse", metrics["rmse"])
        mlflow.log_metric("mae", metrics["mae"])
        mlflow.log_metric("r2", metrics["r2"])
        mlflow.log_metric("tuning_time_seconds", tuning_time)
        mlflow.set_tag("stage", "best_model")
        mlflow.set_tag(
            "mlflow.note.content",
            f"Meilleur modèle après RandomizedSearchCV (8 configurations x 3 folds) sur {best_baseline_name}.",
        )

        signature = mlflow.models.infer_signature(X_train, best_pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            best_pipeline,
            name="model",
            signature=signature,
            input_example=X_train.head(3),
            registered_model_name=None,
            serialization_format="cloudpickle",
        )

        best_run_id = parent_run.info.run_id
        print(f"\n[BEST MODEL] {best_baseline_name} optimisé : RMSE={metrics['rmse']:.4f} R2={metrics['r2']:.4f}")
        print("Meilleurs hyperparamètres :", search.best_params_)

    # --- Sauvegarde locale pour l'API (indépendante d'un serveur MLflow) ---
    joblib.dump(best_pipeline, MODELS_DIR / "model.pkl")

    _, df_world = load_raw_datasets()
    df_world_clean = clean_world_dataset(df_world)
    benchmarks = build_crop_benchmarks(df_world_clean)
    benchmarks.to_csv(MODELS_DIR / "crop_benchmarks.csv", index=False)

    metadata = {
        "best_model_type": best_baseline_name,
        "best_params": search.best_params_,
        "metrics": metrics,
        "mlflow_run_id": best_run_id,
        "mlflow_experiment": MLFLOW_EXPERIMENT,
        "feature_columns": feature_cols,
        "target_column": TARGET_COLUMN,
        "valid_crops": ALL_CROPS,
        "valid_regions": ALL_REGIONS,
        "valid_soil_types": ALL_SOIL_TYPES,
        "valid_weather": ALL_WEATHER,
        "trained_on_rows": len(df_sample),
        "trained_on_total_dataset_rows": n_total,
        "seed": SEED,
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    results_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    print("\nModèle, benchmarks et métadonnées sauvegardés dans models/")
    print("Comparatif des modèles sauvegardé dans reports/model_comparison.csv")


if __name__ == "__main__":
    main()
