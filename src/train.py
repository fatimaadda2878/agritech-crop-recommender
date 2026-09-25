"""
Entraînement, comparaison et optimisation des modèles de prédiction du
rendement agricole, avec suivi complet des expérimentations dans MLflow.

Protocole d'évaluation (pour éviter de choisir et d'évaluer un modèle sur
les mêmes données) :

1. L'échantillon est séparé une seule fois en un jeu d'entraînement (80 %)
   et un jeu de test (20 %). Le jeu de test est mis de côté.
2. Les modèles candidats sont comparés par validation croisée à 5 plis sur
   le jeu d'entraînement uniquement.
3. Les hyperparamètres du meilleur candidat sont recherchés, eux aussi, par
   validation croisée sur le jeu d'entraînement.
4. Le modèle retenu est réentraîné sur tout le jeu d'entraînement, puis
   évalué une seule fois sur le jeu de test. Ce score final n'a servi à
   aucun choix.

Le modèle final est enregistré dans models/ puis copié automatiquement dans
api/models/, le dossier servi par l'API.

Usage:
    python src/train.py
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, train_test_split
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
API_MODELS_DIR = Path("api/models")
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

# Nombre de plis de la validation croisée (comparaison et optimisation)
CV_FOLDS = 5

# Gain minimal de RMSE en validation croisée (en t/ha) pour préférer le
# modèle optimisé au modèle de base. En dessous, l'optimisation n'apporte
# pas d'amélioration réelle et on garde le modèle le plus simple.
MIN_RMSE_GAIN = 0.001

MLFLOW_EXPERIMENT = "crop_yield_prediction"

CV_SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


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


def make_pipeline(model) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor()), ("model", model)])


def load_training_data() -> pd.DataFrame:
    csv_path = DATA_PROCESSED / "merged_dataset.csv"
    if csv_path.exists():
        print(f"Chargement du dataset consolidé existant : {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print("Dataset consolidé introuvable, reconstruction depuis les données brutes...")
        df, _ = build_merged_dataset()
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
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


def summarize_cv(cv_results: dict) -> dict:
    """Moyenne et écart-type des scores de validation croisée (RMSE/MAE positifs)."""
    rmse = -cv_results["test_rmse"]
    mae = -cv_results["test_mae"]
    r2 = cv_results["test_r2"]
    return {
        "cv_rmse_mean": float(rmse.mean()),
        "cv_rmse_std": float(rmse.std()),
        "cv_mae_mean": float(mae.mean()),
        "cv_r2_mean": float(r2.mean()),
    }


def log_dataset_context(n_total: int, n_sample: int, n_train: int, n_test: int) -> None:
    mlflow.log_param("dataset_total_rows", n_total)
    mlflow.log_param("dataset_sample_rows", n_sample)
    mlflow.log_param("train_rows", n_train)
    mlflow.log_param("test_rows", n_test)
    mlflow.log_param("cv_folds", CV_FOLDS)
    mlflow.log_param("random_seed", SEED)


def tuning_space(best_baseline_name: str):
    """Modèle et grille d'hyperparamètres à explorer pour le meilleur candidat."""
    if best_baseline_name == "random_forest":
        return "random_forest_tuned", RandomForestRegressor(random_state=SEED, n_jobs=-1), {
            "model__n_estimators": [100, 150, 200],
            "model__max_depth": [8, 12, 16, None],
            "model__min_samples_leaf": [1, 2, 5],
        }
    if best_baseline_name == "hist_gradient_boosting":
        return "hist_gradient_boosting_tuned", HistGradientBoostingRegressor(random_state=SEED), {
            "model__max_iter": [150, 200, 300],
            "model__max_depth": [6, 8, 10, None],
            "model__learning_rate": [0.03, 0.06, 0.1],
            "model__l2_regularization": [0.0, 0.1, 0.5],
        }
    # La régression linéaire simple n'a pas d'hyperparamètre à optimiser : on
    # teste sa variante régularisée (Ridge) pour vérifier si une
    # régularisation améliore les performances.
    return "ridge_regression", Ridge(random_state=SEED), {
        "model__alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
        "model__solver": ["auto", "svd", "cholesky"],
    }


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df = load_training_data()
    n_total = len(df)
    df_sample = sample_data(df, SAMPLE_SIZE)
    print(f"Dataset total: {n_total} lignes | échantillon: {len(df_sample)} lignes")

    feature_cols = FEATURE_COLUMNS_NUMERIC[:-1] + FEATURE_COLUMNS_BOOL + FEATURE_COLUMNS_CATEGORICAL
    X = df_sample[feature_cols]
    y = df_sample[TARGET_COLUMN]

    # Séparation unique : le jeu de test n'est plus utilisé avant l'évaluation finale.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=df_sample["Crop"]
    )
    n_train, n_test = len(X_train), len(X_test)
    print(f"Train: {X_train.shape} | Test (mis de côté): {X_test.shape}")

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    # ------------------------------------------------------------------
    # 1) Comparaison des modèles candidats par validation croisée
    # ------------------------------------------------------------------
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
        with mlflow.start_run(run_name=f"cv_{name}") as run:
            t0 = time.time()
            cv_results = cross_validate(make_pipeline(model), X_train, y_train, cv=cv, scoring=CV_SCORING)
            cv_time = time.time() - t0
            scores = summarize_cv(cv_results)

            mlflow.log_param("model_type", name)
            mlflow.log_params({f"hp_{k}": v for k, v in model.get_params().items() if not callable(v)})
            log_dataset_context(n_total, len(df_sample), n_train, n_test)
            mlflow.log_metrics(scores)
            mlflow.log_metric("cv_time_seconds", cv_time)
            mlflow.set_tag("stage", "model_selection")
            mlflow.set_tag(
                "mlflow.note.content",
                f"Modèle candidat {name}, évalué par validation croisée à {CV_FOLDS} plis "
                "sur le jeu d'entraînement (le jeu de test n'est pas utilisé).",
            )

            print(
                f"[cv] {name}: RMSE={scores['cv_rmse_mean']:.4f} ± {scores['cv_rmse_std']:.4f} "
                f"R2={scores['cv_r2_mean']:.4f} ({cv_time:.1f}s)"
            )
            results.append({"name": name, "stage": "comparaison", **scores, "run_id": run.info.run_id})

    results_df = pd.DataFrame(results).sort_values("cv_rmse_mean")
    print("\nClassement des modèles candidats (RMSE moyen en validation croisée) :")
    print(results_df[["name", "cv_rmse_mean", "cv_rmse_std", "cv_r2_mean"]])

    best_baseline = results_df.iloc[0]
    best_baseline_name = best_baseline["name"]

    # ------------------------------------------------------------------
    # 2) Optimisation des hyperparamètres, toujours par validation croisée
    # ------------------------------------------------------------------
    tuned_name, base_model, param_distributions = tuning_space(best_baseline_name)
    print(f"\nMeilleur candidat : {best_baseline_name} -> recherche d'hyperparamètres ({tuned_name})")

    search = RandomizedSearchCV(
        make_pipeline(base_model),
        param_distributions=param_distributions,
        n_iter=8,
        scoring=CV_SCORING,
        refit="rmse",
        cv=cv,
        random_state=SEED,
        n_jobs=1,
    )

    with mlflow.start_run(run_name=f"tuning_{tuned_name}") as tuning_run:
        t0 = time.time()
        search.fit(X_train, y_train)
        tuning_time = time.time() - t0

        for i, params in enumerate(search.cv_results_["params"]):
            with mlflow.start_run(run_name=f"candidate_{i}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("cv_rmse_mean", float(-search.cv_results_["mean_test_rmse"][i]))
                mlflow.log_metric("cv_rmse_std", float(search.cv_results_["std_test_rmse"][i]))
                mlflow.log_metric("cv_r2_mean", float(search.cv_results_["mean_test_r2"][i]))
                mlflow.set_tag("stage", "hyperparameter_search")

        best_idx = search.best_index_
        tuned_scores = {
            "cv_rmse_mean": float(-search.cv_results_["mean_test_rmse"][best_idx]),
            "cv_rmse_std": float(search.cv_results_["std_test_rmse"][best_idx]),
            "cv_mae_mean": float(-search.cv_results_["mean_test_mae"][best_idx]),
            "cv_r2_mean": float(search.cv_results_["mean_test_r2"][best_idx]),
        }
        mlflow.log_param("model_type", tuned_name)
        mlflow.log_params(search.best_params_)
        log_dataset_context(n_total, len(df_sample), n_train, n_test)
        mlflow.log_metrics(tuned_scores)
        mlflow.log_metric("tuning_time_seconds", tuning_time)
        mlflow.set_tag("stage", "hyperparameter_search")
        mlflow.set_tag(
            "mlflow.note.content",
            f"Recherche d'hyperparamètres ({len(search.cv_results_['params'])} configurations x "
            f"{CV_FOLDS} plis) sur le jeu d'entraînement uniquement.",
        )
        tuning_run_id = tuning_run.info.run_id

    results_df = pd.concat(
        [results_df, pd.DataFrame([{"name": tuned_name, "stage": "optimisation", **tuned_scores, "run_id": tuning_run_id}])],
        ignore_index=True,
    )

    gain = float(best_baseline["cv_rmse_mean"]) - tuned_scores["cv_rmse_mean"]
    print(
        f"[tuning] {tuned_name}: RMSE={tuned_scores['cv_rmse_mean']:.4f} "
        f"(gain vs {best_baseline_name} : {gain:+.5f} t/ha) | meilleurs paramètres : {search.best_params_}"
    )

    # ------------------------------------------------------------------
    # 3) Choix du modèle final (sur la validation croisée uniquement)
    # ------------------------------------------------------------------
    if gain >= MIN_RMSE_GAIN:
        final_name = tuned_name
        final_pipeline = make_pipeline(clone(base_model)).set_params(**search.best_params_)
        final_params = search.best_params_
        selection_note = (
            f"L'optimisation améliore le RMSE de validation croisée de {gain:.4f} t/ha : "
            f"le modèle optimisé ({tuned_name}) est retenu."
        )
    else:
        final_name = best_baseline_name
        final_pipeline = make_pipeline(clone(candidates[best_baseline_name]))
        final_params = {}
        selection_note = (
            f"L'optimisation ({tuned_name}) n'améliore pratiquement pas le RMSE de validation "
            f"croisée (gain de {gain:.5f} t/ha, sous le seuil de {MIN_RMSE_GAIN} t/ha) : "
            f"le modèle de base, plus simple ({best_baseline_name}), est retenu."
        )
    print(f"\n{selection_note}")

    # ------------------------------------------------------------------
    # 4) Réentraînement sur tout le jeu d'entraînement + évaluation finale
    #    unique sur le jeu de test
    # ------------------------------------------------------------------
    with mlflow.start_run(run_name=f"final_{final_name}") as final_run:
        t0 = time.time()
        final_pipeline.fit(X_train, y_train)
        train_time = time.time() - t0
        test_metrics = evaluate(y_test, final_pipeline.predict(X_test))

        mlflow.log_param("model_type", final_name)
        if final_params:
            mlflow.log_params(final_params)
        log_dataset_context(n_total, len(df_sample), n_train, n_test)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metric("train_time_seconds", train_time)
        mlflow.set_tag("stage", "final_model")
        mlflow.set_tag(
            "mlflow.note.content",
            f"{selection_note} Modèle réentraîné sur tout le jeu d'entraînement, puis évalué "
            "une seule fois sur le jeu de test mis de côté.",
        )

        signature = mlflow.models.infer_signature(X_train, final_pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            final_pipeline,
            name="model",
            signature=signature,
            input_example=X_train.head(3),
            registered_model_name=None,
            serialization_format="cloudpickle",
        )
        final_run_id = final_run.info.run_id

    print(
        f"[FINAL] {final_name} — évaluation unique sur le test : RMSE={test_metrics['rmse']:.4f} "
        f"MAE={test_metrics['mae']:.4f} R2={test_metrics['r2']:.4f}"
    )

    # ------------------------------------------------------------------
    # 5) Sauvegarde locale (indépendante d'un serveur MLflow) + copie dans l'API
    # ------------------------------------------------------------------
    joblib.dump(final_pipeline, MODELS_DIR / "model.pkl")

    _, df_world = load_raw_datasets()
    df_world_clean = clean_world_dataset(df_world)
    benchmarks = build_crop_benchmarks(df_world_clean)
    benchmarks.to_csv(MODELS_DIR / "crop_benchmarks.csv", index=False)

    final_cv = (
        tuned_scores if final_name == tuned_name
        else results_df.loc[results_df["name"] == final_name].iloc[0][list(tuned_scores)].to_dict()
    )
    metadata = {
        "best_model_type": final_name,
        "best_params": final_params,
        "selection_note": selection_note,
        # Métriques de l'évaluation finale, obtenues une seule fois sur le jeu de test
        "metrics": test_metrics,
        "cv_metrics": {k: float(v) for k, v in final_cv.items()},
        "evaluation_protocol": (
            f"Comparaison et optimisation par validation croisée à {CV_FOLDS} plis sur le jeu "
            "d'entraînement (80 %), évaluation finale unique sur le jeu de test (20 %)."
        ),
        "mlflow_run_id": final_run_id,
        "mlflow_experiment": MLFLOW_EXPERIMENT,
        "feature_columns": feature_cols,
        "target_column": TARGET_COLUMN,
        "valid_crops": ALL_CROPS,
        "valid_regions": ALL_REGIONS,
        "valid_soil_types": ALL_SOIL_TYPES,
        "valid_weather": ALL_WEATHER,
        "trained_on_rows": n_train,
        "test_rows": n_test,
        "trained_on_total_dataset_rows": n_total,
        "seed": SEED,
    }
    with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    results_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    # Copie automatique des artefacts dans le dossier servi par l'API
    API_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for filename in ("model.pkl", "crop_benchmarks.csv", "model_metadata.json"):
        shutil.copy2(MODELS_DIR / filename, API_MODELS_DIR / filename)

    print("\nModèle, benchmarks et métadonnées sauvegardés dans models/ et copiés dans api/models/")
    print("Comparatif des modèles sauvegardé dans reports/model_comparison.csv")


if __name__ == "__main__":
    main()
