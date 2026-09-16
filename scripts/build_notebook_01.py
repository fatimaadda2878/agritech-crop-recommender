"""Construit le notebook 01_data_merging_eda.ipynb à partir de cellules
définies ici, puis il est exécuté via nbconvert (voir run_notebooks.sh)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md("""\
# Agritech Answers — Fusion des données & Analyse en Composantes Principales (ACP)

**Mission** : construire le dataset consolidé qui servira à entraîner le moteur de
prédiction/recommandation de rendement agricole.

**Sources de données :**
1. `Agriculture CropYield Dataset` (`crop_yield.csv`) — 1 000 000 lignes, niveau **parcelle**
   (température, pluviométrie, sol, engrais, irrigation, rendement observé).
2. `CropYield Prediction Dataset` (`yield_df.csv`, dérivé de données FAO) — 28 242 lignes,
   niveau **pays / année / culture** (pluviométrie moyenne, température moyenne, usage de
   pesticides, rendement).

**Objectifs de ce notebook :**
- Explorer chaque source (types, valeurs manquantes, distributions, doublons).
- Définir et justifier une stratégie de fusion.
- Produire un dataset consolidé et nettoyé (`data/processed/merged_dataset.csv`).
- Réaliser une ACP sur le dataset parcelle pour identifier les variables clés du rendement.
""")

code("""\
import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.data_prep import (
    load_raw_datasets, clean_parcels_dataset, clean_world_dataset,
    build_crop_benchmarks, merge_datasets, build_merged_dataset,
    CROP_NAME_MAPPING, CROPS_WITHOUT_FAO_MATCH,
)

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 30)
plt.rcParams["figure.dpi"] = 100

RNG = 42
""")

md("## 1. Chargement des données brutes")

code("""\
df_parcels, df_world = load_raw_datasets("../data/raw")
print("Agriculture CropYield Dataset :", df_parcels.shape)
print("CropYield Prediction Dataset  :", df_world.shape)
""")

md("""\
## 2. Exploration — Agriculture CropYield Dataset (niveau parcelle)

C'est notre dataset **cible** : il contient la variable à prédire
(`Yield_tons_per_hectare`) et sera enrichi avec les données de l'autre source.
""")

code("""\
df_parcels.info()
""")

code("""\
df_parcels.describe()
""")

code("""\
print("Valeurs manquantes :")
print(df_parcels.isna().sum())
print("\\nDoublons :", df_parcels.duplicated().sum())
print("\\nValeurs négatives de rendement (physiquement impossibles) :",
      (df_parcels['Yield_tons_per_hectare'] < 0).sum())
""")

md("""\
**Constats :**
- Aucune valeur manquante.
- Aucun doublon.
- 231 lignes ont un rendement négatif (bruit de simulation) : nous les ramènerons à 0
  plutôt que de les supprimer, car les autres variables de ces lignes restent informatives.
- Pas de colonne pays / année dans ce dataset : seule une variable `Region`
  (North / South / East / West) est disponible, ce qui **exclut une jointure directe
  par pays et année** avec le second dataset.
""")

code("""\
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
num_cols = ["Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest", "Yield_tons_per_hectare"]
sample = df_parcels.sample(50_000, random_state=RNG)

for ax, col in zip(axes.flat, num_cols):
    sns.histplot(sample[col], kde=True, ax=ax, color="#3b7a57")
    ax.set_title(f"Distribution — {col}")

sns.countplot(data=sample, x="Crop", ax=axes.flat[4], color="#3b7a57")
axes.flat[4].set_title("Répartition des cultures")
axes.flat[4].tick_params(axis="x", rotation=30)

sns.countplot(data=sample, x="Region", ax=axes.flat[5], color="#3b7a57")
axes.flat[5].set_title("Répartition des régions")

plt.tight_layout()
plt.savefig("../reports/fig_distributions_parcelles.png", bbox_inches="tight")
plt.show()
""")

code("""\
plt.figure(figsize=(6, 5))
corr = df_parcels[num_cols].corr()
sns.heatmap(corr, annot=True, cmap="RdBu_r", vmin=-1, vmax=1, center=0)
plt.title("Corrélations — variables numériques du dataset parcelle")
plt.tight_layout()
plt.savefig("../reports/fig_correlations_parcelles.png", bbox_inches="tight")
plt.show()
""")

md("""\
Les corrélations linéaires entre variables numériques brutes et le rendement sont
**faibles** : cela suggère que le rendement dépend surtout des variables catégorielles
(culture, sol, irrigation, engrais) et/ou d'interactions non-linéaires — ce que
l'ACP et les modèles non-linéaires exploreront plus loin.
""")

md("""\
## 3. Exploration — CropYield Prediction Dataset (niveau pays / année)

Ce second dataset (source FAO) fournit des indicateurs climatiques et agronomiques
réels, mais à une granularité différente (pays × année × culture, et non parcelle).
""")

code("""\
df_world.info()
""")

code("""\
print("Doublons :", df_world.duplicated().sum())
print("Période couverte :", df_world['Year'].min(), "-", df_world['Year'].max())
print("Nombre de pays :", df_world['Area'].nunique())
print("Cultures couvertes :", sorted(df_world['Item'].unique()))
""")

md("""\
**Constat important :** 2 310 lignes strictement dupliquées sont présentes — elles
seront supprimées. Par ailleurs, la variable `pesticides_tonnes` est en réalité un
**total national de pesticides tous produits confondus** (et non un usage spécifique
à la culture) : c'est une limitation à garder en tête dans l'interprétation.
""")

code("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["average_rain_fall_mm_per_year", "avg_temp", "pesticides_tonnes"]):
    sns.histplot(df_world[col], kde=True, ax=ax, color="#c98a2c")
    ax.set_title(col)
plt.tight_layout()
plt.savefig("../reports/fig_distributions_monde.png", bbox_inches="tight")
plt.show()
""")

md("""\
## 4. Stratégie de fusion

Les deux datasets **ne partagent pas de clé pays/année** (le dataset parcelle n'a ni
pays ni année). Une jointure directe pays+année est donc impossible.

**Stratégie retenue :** utiliser la **culture (`Crop` / `Item`)** comme clé de
jointure, en agrégeant le CropYield Prediction Dataset par culture (moyenne sur tous
pays et toutes années 1990-2013) pour obtenir des **valeurs de référence mondiales**
par culture : pluviométrie moyenne, température moyenne, usage moyen de pesticides,
rendement moyen mondial.

Ces indicateurs sont ensuite fusionnés sur chaque ligne du dataset parcelle
correspondant à la même culture, et utilisés pour calculer des **variables d'écart**
(proxy) :
- `rainfall_gap_pct` : écart en % entre la pluviométrie de la parcelle et la moyenne
  mondiale pour cette culture ;
- `temperature_gap_c` : écart en °C entre la température de la parcelle et la moyenne
  mondiale pour cette culture.

**Correspondance des noms de cultures** (les deux sources n'utilisent pas toujours le
même nom) :
""")

code("""\
print(CROP_NAME_MAPPING)
print("\\nCultures du dataset parcelle sans correspondance FAO directe :", CROPS_WITHOUT_FAO_MATCH)
""")

md("""\
`Barley` (orge) et `Cotton` (coton) n'ont pas d'équivalent dans le CropYield
Prediction Dataset. **Limitation documentée** : pour ces deux cultures, on applique la
moyenne mondiale *tous produits confondus* comme valeur de repli plutôt que de
supprimer ces cultures (qui représentent tout de même ~1/3 des lignes du dataset
parcelle).
""")

code("""\
df_parcels_clean = clean_parcels_dataset(df_parcels)
df_world_clean = clean_world_dataset(df_world)
benchmarks = build_crop_benchmarks(df_world_clean)
benchmarks
""")

code("""\
merged = merge_datasets(df_parcels_clean, benchmarks)
print(merged.shape)
merged.head()
""")

code("""\
print("Valeurs manquantes après fusion :", merged.isna().sum().sum())
print("Lignes avec rendement négatif corrigé à 0 :", df_parcels_clean.attrs['n_negative_yield_clipped'])
""")

md("""\
## 5. Analyse en Composantes Principales (ACP)

Objectif : identifier, parmi toutes les variables disponibles (initiales + variables
d'écart créées lors de la fusion), celles qui expliquent le plus la variabilité du
rendement — et donc les variables clés à surveiller pour un agriculteur.

On standardise (centrage-réduction) toutes les variables numériques et on encode les
variables catégorielles en indicatrices avant de lancer l'ACP.
""")

code("""\
from sklearn.preprocessing import OneHotEncoder

numeric_features = [
    "Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest",
    "world_avg_rainfall_mm", "world_avg_temp_c", "world_avg_pesticides_tonnes",
    "world_avg_yield_t_ha", "rainfall_gap_pct", "temperature_gap_c",
    "Yield_tons_per_hectare",
]
bool_features = ["Fertilizer_Used", "Irrigation_Used"]

pca_sample = merged.sample(20_000, random_state=RNG).reset_index(drop=True)

X_num = pca_sample[numeric_features].astype(float)
X_bool = pca_sample[bool_features].astype(int)
X_pca_input = pd.concat([X_num, X_bool], axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca_input)

pca = PCA(random_state=RNG)
pca_coords = pca.fit_transform(X_scaled)

print("Variance expliquée par composante :", pca.explained_variance_ratio_.round(3))
""")

md("### 5.1 Choix du nombre de composantes (scree plot)")

code("""\
explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.bar(range(1, len(explained) + 1), explained, color="#3b7a57", alpha=0.8, label="Variance expliquée")
ax1.set_xlabel("Composante principale")
ax1.set_ylabel("Variance expliquée")
ax2 = ax1.twinx()
ax2.plot(range(1, len(explained) + 1), cumulative, color="#c9432c", marker="o", label="Cumulée")
ax2.set_ylabel("Variance cumulée")
ax2.axhline(0.8, color="grey", linestyle="--", linewidth=1)
fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
plt.title("Scree plot — ACP dataset parcelle")
plt.tight_layout()
plt.savefig("../reports/fig_pca_scree.png", bbox_inches="tight")
plt.show()

print(f"F1+F2 expliquent {cumulative[1]*100:.1f}% de la variance totale")
""")

md("""\
### 5.2 Cercle des corrélations (F1, F2)

Le cercle des corrélations montre la contribution de chaque variable initiale aux
deux premiers axes principaux. Une variable proche du bord du cercle est bien
représentée sur ce plan ; deux variables proches l'une de l'autre sont corrélées
positivement, à l'opposé négativement, à 90° non corrélées.
""")

code("""\
loadings = pca.components_[:2].T * np.sqrt(pca.explained_variance_[:2])

fig, ax = plt.subplots(figsize=(7, 7))
circle = plt.Circle((0, 0), 1, facecolor="none", edgecolor="grey", linestyle="--")
ax.add_patch(circle)

for i, feature in enumerate(X_pca_input.columns):
    ax.arrow(0, 0, loadings[i, 0], loadings[i, 1], head_width=0.02, color="#c9432c", alpha=0.8)
    ax.text(loadings[i, 0] * 1.12, loadings[i, 1] * 1.12, feature, fontsize=9, ha="center")

ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.axhline(0, color="grey", linewidth=0.8)
ax.axvline(0, color="grey", linewidth=0.8)
ax.set_xlabel(f"F1 ({explained[0]*100:.1f}%)")
ax.set_ylabel(f"F2 ({explained[1]*100:.1f}%)")
ax.set_title("Cercle des corrélations — F1 x F2")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("../reports/fig_pca_cercle_correlations.png", bbox_inches="tight")
plt.show()
""")

code("""\
contrib = pd.DataFrame(
    {"F1": pca.components_[0], "F2": pca.components_[1]},
    index=X_pca_input.columns,
).abs().sort_values("F1", ascending=False)
contrib
""")

md("""\
### 5.3 Projection des individus sur le premier plan factoriel
""")

code("""\
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(
    pca_coords[:, 0], pca_coords[:, 1],
    c=pca_sample["Yield_tons_per_hectare"], cmap="viridis", s=6, alpha=0.5,
)
plt.colorbar(scatter, label="Rendement (t/ha)")
ax.set_xlabel(f"F1 ({explained[0]*100:.1f}%)")
ax.set_ylabel(f"F2 ({explained[1]*100:.1f}%)")
ax.set_title("Projection des individus — coloration par rendement")
plt.tight_layout()
plt.savefig("../reports/fig_pca_individus.png", bbox_inches="tight")
plt.show()
""")

md("""\
### 5.4 Interprétation — variables clés identifiées par l'ACP

- **F1** est structuré principalement par les variables *monde* liées à la culture
  (`world_avg_pesticides_tonnes`, `world_avg_rainfall_mm`, `world_avg_temp_c`,
  `world_avg_yield_t_ha`) : cet axe oppose surtout les cultures entre elles
  (leur profil climatique et agronomique moyen), plus qu'il ne discrimine le
  rendement au sein d'une même culture.
- **F2** capte davantage les variables *parcelle* (`Rainfall_mm`,
  `Temperature_Celsius`, écarts à la moyenne mondiale) et se rapproche le plus de la
  direction du rendement observé.
- Le nuage de points projeté ne montre **pas de séparation nette** du rendement selon
  F1/F2 seuls : cela confirme l'intuition vue en partie 2 (corrélations linéaires
  faibles) — le rendement dépend d'une **combinaison non-linéaire** de variables
  (notamment `Irrigation_Used`, `Fertilizer_Used`, `Soil_Type`, qui ne sont pas
  linéairement séparables en 2D). Cela justifie le choix, à l'étape suivante, de
  modèles non-linéaires (Random Forest / Gradient Boosting) plutôt qu'une simple
  régression linéaire.
- Variables clés retenues pour le modèle : `Crop`, `Region`, `Soil_Type`,
  `Irrigation_Used`, `Fertilizer_Used`, `Rainfall_mm`, `Temperature_Celsius`,
  `Days_to_Harvest`, ainsi que les variables d'écart mondiales créées lors de la
  fusion.
""")

md("""\
## 6. Export du dataset consolidé
""")

code("""\
merged_full, stats = build_merged_dataset("../data/raw")
print(stats)

import os
os.makedirs("../data/processed", exist_ok=True)
merged_full.to_csv("../data/processed/merged_dataset.csv", index=False)
print("Export terminé :", merged_full.shape)
""")

md("""\
## 7. Résumé des choix (pour le rapport métier)

- **Fusion** : jointure par culture (et non pays/année, absent du dataset parcelle),
  avec agrégation du CropYield Prediction Dataset en benchmarks mondiaux par culture.
- **Nettoyage** : suppression de 2 310 doublons dans le CropYield Prediction Dataset ;
  correction de 231 rendements négatifs (mis à 0) dans le dataset parcelle ; aucune
  valeur manquante après fusion.
- **Limitation documentée** : Barley et Cotton n'ont pas de correspondance FAO directe
  → moyenne mondiale tous produits confondus utilisée en repli. Les pesticides FAO
  sont un total national, pas spécifique à la culture.
- **Variables clés identifiées (ACP + corrélations)** : la culture elle-même, le mode
  d'irrigation et d'engrais, le sol, ainsi que les écarts de pluviométrie/température
  par rapport aux normales mondiales de la culture.
- **Dataset final** : `data/processed/merged_dataset.csv`, 1 000 000 lignes, 0 valeur
  manquante, prêt pour l'entraînement des modèles.
""")

nb["cells"] = cells
nbf.write(nb, "notebooks/01_data_merging_eda.ipynb")
print("Notebook écrit : notebooks/01_data_merging_eda.ipynb")
