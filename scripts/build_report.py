"""Génère le rapport métier PDF (accessible à un public non technique)."""
import json

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

GREEN = colors.HexColor("#3b7a57")
DARK = colors.HexColor("#1f2d24")
GREY = colors.HexColor("#5a5a5a")
LIGHT_BG = colors.HexColor("#eef5f0")

import csv

# Fichiers versionnés dans le dépôt (écrits par src/train.py)
with open("api/models/model_metadata.json", encoding="utf-8") as f:
    metadata = json.load(f)
with open("reports/model_comparison.csv", encoding="utf-8") as f:
    comparison = {row["name"]: row for row in csv.DictReader(f)}

metrics = metadata["metrics"]          # évaluation finale unique sur le jeu de test
cv_metrics = metadata["cv_metrics"]    # validation croisée du modèle retenu


def fr(x, nd=3):
    """Nombre au format français."""
    return f"{float(x):.{nd}f}".replace(".", ",")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("CoverTitle", fontSize=28, leading=34, textColor=DARK, fontName="Helvetica-Bold", spaceAfter=10))
styles.add(ParagraphStyle("CoverSubtitle", fontSize=15, leading=20, textColor=GREEN, fontName="Helvetica", spaceAfter=6))
styles.add(ParagraphStyle("CoverMeta", fontSize=10.5, leading=15, textColor=GREY))
styles.add(ParagraphStyle("H1", fontSize=17, leading=21, textColor=DARK, fontName="Helvetica-Bold", spaceBefore=18, spaceAfter=10))
styles.add(ParagraphStyle("H2", fontSize=13, leading=17, textColor=GREEN, fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=6))
styles.add(ParagraphStyle("Body", fontSize=10.3, leading=15, textColor=DARK, spaceAfter=8, alignment=4))
styles.add(ParagraphStyle("BodyBold", parent=styles["Body"], fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("Caption", fontSize=9, leading=12, textColor=GREY, alignment=1, spaceAfter=14, fontName="Helvetica-Oblique"))
styles.add(ParagraphStyle("KpiLabel", fontSize=9.5, leading=12, textColor=GREY, alignment=1))
styles.add(ParagraphStyle("KpiValue", fontSize=20, leading=24, textColor=GREEN, fontName="Helvetica-Bold", alignment=1, spaceAfter=2))
styles.add(ParagraphStyle("BulletItem", parent=styles["Body"], spaceAfter=4))

story = []

# ---------------------------------------------------------------- COVER ----
story.append(Spacer(1, 6 * cm))
accent_bar = Table([[""]], colWidths=[3 * cm], rowHeights=[0.25 * cm])
accent_bar.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), GREEN)]))
story.append(accent_bar)
story.append(Spacer(1, 0.8 * cm))
story.append(Paragraph("Optimiser le choix des cultures", styles["CoverTitle"]))
story.append(Paragraph("et prédire le rendement agricole", styles["CoverTitle"]))
story.append(Spacer(1, 0.3 * cm))
story.append(Paragraph(
    "Rapport métier — Moteur de prédiction &amp; de recommandation de rendement",
    styles["CoverSubtitle"],
))
story.append(Spacer(1, 3 * cm))
story.append(Paragraph("Fatima Adda — Data Scientist, Agritech Answers", styles["CoverMeta"]))
story.append(Paragraph("Septembre 2026", styles["CoverMeta"]))
story.append(Paragraph(
    "Document destiné à un public non technique : direction, équipes commerciales, agriculteurs partenaires.",
    styles["CoverMeta"],
))
story.append(PageBreak())

# ---------------------------------------------------------- EXEC SUMMARY ---
story.append(Paragraph("Résumé en un coup d'œil", styles["H1"]))
story.append(Paragraph(
    "J'ai construit un modèle qui estime le rendement (en tonnes par hectare) d'une "
    "parcelle agricole à partir de ses conditions (région, sol, météo, pluviométrie, "
    "température, irrigation, engrais) et de la culture choisie. Ce modèle alimente une "
    "application où l'agriculteur peut soit <b>obtenir une estimation de rendement</b> pour "
    "une culture qu'il a déjà choisie, soit <b>obtenir un classement indicatif des 6 "
    "cultures possibles</b> selon leur rendement estimé pour les conditions de sa parcelle. "
    "Ce classement porte sur le rendement, pas sur la rentabilité (prix de vente et coûts "
    "de production ne sont pas pris en compte).",
    styles["Body"],
))

kpi_data = [
    [
        Paragraph(f"{metrics['r2']*100:.0f}%", styles["KpiValue"]),
        Paragraph(f"{metrics['rmse']:.2f} t/ha".replace(".", ","), styles["KpiValue"]),
        Paragraph("1 000 000", styles["KpiValue"]),
        Paragraph("6", styles["KpiValue"]),
    ],
    [
        Paragraph("de la variabilité du<br/>rendement est expliquée<br/>par le modèle", styles["KpiLabel"]),
        Paragraph("d'erreur moyenne<br/>sur l'estimation<br/>de rendement", styles["KpiLabel"]),
        Paragraph("parcelles analysées<br/>pour entraîner<br/>le modèle", styles["KpiLabel"]),
        Paragraph("cultures couvertes<br/>par le moteur de<br/>recommandation", styles["KpiLabel"]),
    ],
]
kpi_table = Table(kpi_data, colWidths=[4.1 * cm] * 4)
kpi_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cfe0d6")),
    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.white),
    ("TOPPADDING", (0, 0), (-1, 0), 12),
    ("BOTTOMPADDING", (0, -1), (-1, -1), 12),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
]))
story.append(Spacer(1, 6))
story.append(kpi_table)
story.append(Spacer(1, 14))

story.append(Paragraph("Le message clé pour les agriculteurs", styles["H2"]))
story.append(Paragraph(
    "L'analyse des données montre que, à conditions climatiques égales, <b>l'irrigation et "
    "l'usage d'engrais ont un impact bien plus important sur le rendement que le choix de la "
    "culture elle-même</b>. Une parcelle irriguée et fertilisée produit en moyenne "
    "<b>6,00 t/ha</b>, contre <b>3,30 t/ha</b> pour une parcelle ni irriguée ni fertilisée — "
    "soit près de <b>2 fois plus</b>. À l'inverse, les 6 cultures étudiées (blé, orge, coton, "
    "maïs, riz, soja) affichent des rendements moyens très proches (entre 4,64 et 4,65 t/ha) "
    "sur l'ensemble des parcelles. <b>Le premier levier d'action recommandé aux agriculteurs "
    "est donc l'irrigation et la fertilisation, avant le choix de la culture.</b>",
    styles["Body"],
))
story.append(PageBreak())

# ------------------------------------------------------------- DEMARCHE ----
story.append(Paragraph("Comment j'ai construit ce modèle", styles["H1"]))
story.append(Paragraph(
    "J'ai combiné deux sources de données complémentaires :",
    styles["Body"],
))
story.append(ListFlowable([
    ListItem(Paragraph(
        "<b>Des données de parcelles</b> (1 million de lignes) : pour chaque parcelle, la "
        "région, le type de sol, la culture, la pluviométrie, la température, l'usage "
        "d'engrais et d'irrigation, la météo, le nombre de jours avant récolte, et le "
        "rendement réellement obtenu.", styles["BulletItem"])),
    ListItem(Paragraph(
        "<b>Des données climatiques et agronomiques mondiales</b> (source FAO, 1990-2013, "
        "101 pays) : pour chaque culture, la pluviométrie moyenne, la température moyenne et "
        "l'usage de pesticides typiques dans le monde.", styles["BulletItem"])),
], bulletType="bullet", start="•"))

story.append(Paragraph(
    "Ces deux sources ne pouvaient pas être combinées directement (le premier jeu de données "
    "n'indique ni le pays ni l'année). J'ai donc calculé, pour chaque culture, des "
    "<b>valeurs de référence mondiales</b> (pluviométrie, température, pesticides moyens) à "
    "partir des données FAO, puis je les ai associées à chaque parcelle selon sa culture. "
    "Cela permet de savoir si une parcelle est plus ou moins arrosée/chaude que la "
    "moyenne mondiale pour sa culture — une information utile pour le modèle. Deux cultures "
    "(orge et coton) n'ayant pas d'équivalent dans les données mondiales disponibles, je "
    "leur ai appliqué la moyenne mondiale tous produits confondus, une limitation "
    "documentée dans le notebook technique.",
    styles["Body"],
))

story.append(Paragraph(
    "J'ai aussi vérifié la qualité des données : 231 valeurs de rendement négatives "
    "(impossibles physiquement, dues à du bruit de mesure/simulation) ont été ramenées à 0, "
    "et 2 310 lignes dupliquées ont été supprimées des données mondiales. Après ce nettoyage, "
    "le jeu de données final ne contient aucune valeur manquante.",
    styles["Body"],
))

story.append(PageBreak())
story.append(Paragraph("Identifier les variables qui comptent vraiment (ACP)", styles["H1"]))
story.append(Paragraph(
    "Pour comprendre comment les variables s'organisent, j'ai utilisé une méthode "
    "statistique appelée <b>analyse en composantes principales</b> (ACP). Elle résume "
    "les nombreuses variables décrivant une parcelle en quelques grands « axes » qui "
    "regroupent les variables liées entre elles. Le rendement, qui est ce que l'on "
    "cherche à expliquer, n'a volontairement <b>pas</b> servi à construire ces axes : "
    "il a été placé ensuite sur les graphiques, pour voir de quels axes il est proche "
    "sans qu'il influence leur construction.",
    styles["Body"],
))

img1 = Image("reports/fig_pca_cercle_correlations.png", width=7.4 * cm, height=7.4 * cm)
img2 = Image("reports/fig_pca_cercle_F3_F5.png", width=7.4 * cm, height=7.4 * cm)
tbl_imgs = Table([[img1, img2]], colWidths=[7.9 * cm, 7.9 * cm])
tbl_imgs.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
story.append(Spacer(1, 4))
story.append(KeepTogether([
    tbl_imgs,
    Paragraph(
        "Chaque flèche rouge représente une variable ; la flèche bleue en pointillés est le "
        "rendement, ajouté après coup. Plus une flèche est longue, mieux la variable est "
        "représentée sur le graphique. À gauche, les deux axes principaux : le profil mondial "
        "de la culture (horizontal) et la température (vertical). Le rendement y est à peine "
        "visible. À droite, l'axe de la pluviométrie (horizontal) et celui de l'irrigation "
        "(vertical), vers lesquels le rendement pointe nettement.",
        styles["Caption"],
    ),
    Paragraph(
        "<b>Conclusion de cette analyse :</b> les axes qui résument le plus d'information sur "
        "les parcelles (le profil de la culture, la température) ne sont pas ceux qui "
        "expliquent le rendement. Le rendement est surtout lié à la <b>pluviométrie</b> de la "
        "parcelle, puis à l'<b>irrigation</b> et à l'<b>engrais</b>. Le choix de la culture "
        "compte très peu.",
        styles["Body"],
    ),
]))

# ------------------------------------------------------- VARIABLES CLES ----
story.append(PageBreak())
story.append(Paragraph("Les variables clés expliquées", styles["H1"]))
var_table_data = [
    [Paragraph("<b>Variable</b>", styles["BodyBold"]), Paragraph("<b>Ce que c'est</b>", styles["BodyBold"]), Paragraph("<b>Impact observé</b>", styles["BodyBold"])],
    [Paragraph("Engrais", styles["Body"]), Paragraph("De l'engrais est-il apporté à la parcelle ?", styles["Body"]), Paragraph("Très fort : +1,50 t/ha en moyenne", styles["Body"])],
    [Paragraph("Irrigation", styles["Body"]), Paragraph("La parcelle est-elle irriguée artificiellement ?", styles["Body"]), Paragraph("Très fort : +1,20 t/ha en moyenne", styles["Body"])],
    [Paragraph("Pluviométrie", styles["Body"]), Paragraph("Quantité de pluie reçue par la parcelle sur l'année (100 à 1 000 mm)", styles["Body"]), Paragraph("Fort : environ +0,5 t/ha par tranche de 100 mm supplémentaires", styles["Body"])],
    [Paragraph("Température", styles["Body"]), Paragraph("Température moyenne de la parcelle (15 à 40 °C)", styles["Body"]), Paragraph("Faible : environ +0,2 t/ha pour 10 °C de plus", styles["Body"])],
    [Paragraph("Culture", styles["Body"]), Paragraph("Blé, orge, coton, maïs, riz ou soja", styles["Body"]), Paragraph("Faible sur ce jeu de données (écart &lt; 0,02 t/ha entre cultures)", styles["Body"])],
    [Paragraph("Type de sol", styles["Body"]), Paragraph("Sableux, argileux, limoneux, silto-argileux, tourbeux, calcaire", styles["Body"]), Paragraph("Faible (écart &lt; 0,01 t/ha)", styles["Body"])],
    [Paragraph("Jours avant récolte", styles["Body"]), Paragraph("Durée du cycle de culture jusqu'à la récolte", styles["Body"]), Paragraph("Négligeable", styles["Body"])],
    [Paragraph("Écarts à la moyenne mondiale", styles["Body"]), Paragraph("Différence entre la pluie (ou la température) de la parcelle et la moyenne mondiale pour sa culture", styles["Body"]), Paragraph("Pas d'effet propre : ils reprennent l'information de la pluviométrie et de la température de la parcelle", styles["Body"])],
]
var_table = Table(var_table_data, colWidths=[3.3 * cm, 7.2 * cm, 6.2 * cm], repeatRows=1)
var_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), GREEN),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
]))
story.append(var_table)
story.append(PageBreak())

# ------------------------------------------------------------- RESULTATS ---
story.append(Paragraph("Résultats du modèle", styles["H1"]))
story.append(Paragraph(
    "J'ai travaillé sur un échantillon de 200 000 parcelles (sur les 1 000 000 disponibles, "
    "ce qui suffit largement pour obtenir des résultats stables). Pour que le score final "
    "soit honnête, j'ai d'abord mis de côté <b>20 % des parcelles</b> (le jeu de test), "
    "auxquelles le modèle n'a jamais eu accès pendant sa mise au point :",
    styles["Body"],
))
story.append(ListFlowable([
    ListItem(Paragraph(
        "<b>Choix du modèle</b> : j'ai comparé 3 approches par <b>validation croisée</b> sur "
        "les 80 % restants. Les données sont découpées en 5 parts ; chaque approche est "
        "entraînée 5 fois sur 4 parts et évaluée sur la cinquième, et l'on fait la moyenne.",
        styles["BulletItem"])),
    ListItem(Paragraph(
        "<b>Réglage</b> : j'ai ensuite testé plusieurs réglages de la meilleure approche, "
        "toujours par validation croisée.", styles["BulletItem"])),
    ListItem(Paragraph(
        "<b>Évaluation finale</b> : le modèle retenu est évalué <b>une seule fois</b> sur les "
        "20 % mis de côté. Ce score n'a servi à aucun choix.", styles["BulletItem"])),
], bulletType="bullet", start="•"))

model_labels = [
    ("linear_regression", "Régression linéaire"),
    ("random_forest", "Forêt aléatoire (Random Forest)"),
    ("hist_gradient_boosting", "Gradient Boosting"),
    ("ridge_regression", "Régression Ridge (réglage de la régression linéaire)"),
]
comp_data = [[
    Paragraph("<b>Modèle</b>", styles["BodyBold"]),
    Paragraph("<b>Erreur moyenne en validation croisée (RMSE, t/ha)</b>", styles["BodyBold"]),
    Paragraph("<b>Variabilité expliquée (R<super>2</super>)</b>", styles["BodyBold"]),
]]
for key, label in model_labels:
    row = comparison[key]
    style = styles["BodyBold"] if key == metadata["best_model_type"] else styles["Body"]
    suffix = " — retenu" if key == metadata["best_model_type"] else ""
    comp_data.append([
        Paragraph(label + suffix, style),
        Paragraph(f"{fr(row['cv_rmse_mean'])} ± {fr(row['cv_rmse_std'])}", style),
        Paragraph(f"{float(row['cv_r2_mean']) * 100:.1f} %".replace(".", ","), style),
    ])
comp_table = Table(comp_data, colWidths=[7 * cm, 5.3 * cm, 4.4 * cm])
best_row = 1 + [k for k, _ in model_labels].index(metadata["best_model_type"])
comp_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), GREEN),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("BACKGROUND", (0, best_row), (-1, best_row), LIGHT_BG),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
]))
story.append(comp_table)
story.append(Spacer(1, 8))
story.append(Paragraph(
    f"<b>Évaluation finale sur le jeu de test (régression linéaire) :</b> erreur moyenne "
    f"(RMSE) de <b>{fr(metrics['rmse'])} t/ha</b>, variabilité expliquée (R<super>2</super>) de "
    f"<b>{fr(metrics['r2'] * 100, 1)} %</b>.",
    styles["Body"],
))
story.append(Paragraph(
    "<b>Comment lire ces chiffres ?</b> Le RMSE indique l'erreur moyenne du modèle : une "
    "valeur de 0,50 t/ha signifie que l'estimation se trompe en moyenne de 0,50 tonne par "
    "hectare (sur des rendements qui varient entre 0 et 10 t/ha environ). Le R<super>2</super> "
    "indique la part des différences de rendement entre parcelles que le modèle parvient à "
    "expliquer : environ 91 % est un très bon score. Le score final sur le test est très "
    "proche de celui de la validation croisée : le modèle se comporte aussi bien sur des "
    "parcelles qu'il n'a jamais vues.",
    styles["Body"],
))
story.append(Paragraph(
    "<b>Quel modèle retenir ?</b> Les 3 approches obtiennent des scores presque identiques : "
    "la relation entre les conditions de la parcelle et le rendement est essentiellement "
    "simple (linéaire), et les modèles plus complexes n'apportent rien. Le réglage testé "
    "ensuite (la régression Ridge) <b>n'améliore pratiquement pas</b> la régression linéaire "
    "de départ. J'ai donc gardé la <b>régression linéaire</b>, le modèle le plus simple, le "
    "plus facile à expliquer et à maintenir.",
    styles["Body"],
))

story.append(PageBreak())
story.append(Paragraph("Suivi des expérimentations avec MLflow", styles["H2"]))
story.append(Paragraph(
    "Chaque étape a été enregistrée automatiquement dans l'outil MLflow : la validation "
    "croisée de chaque modèle comparé, les 8 réglages testés pour la régression Ridge, puis "
    "l'évaluation finale du modèle retenu sur le jeu de test. MLflow conserve l'historique "
    "complet des expérimentations (paramètres utilisés, scores obtenus, modèle associé), "
    "ce qui garantit la traçabilité et la reproductibilité de mes résultats.",
    styles["Body"],
))

for img_path, caption in [
    ("mlflow_screenshots/01_mlflow_experiments.png", "Liste des expériences MLflow : l'expérience « crop_yield_prediction » regroupe l'ensemble des runs de ce projet."),
    ("mlflow_screenshots/02_mlflow_runs_list.png", "Liste des runs : validation croisée des 3 modèles comparés (cv_), recherche de réglages de la régression Ridge (tuning_), puis évaluation finale unique sur le jeu de test (final_, colonnes test_)."),
    ("mlflow_screenshots/03_mlflow_run_detail.png", "Détail du run final : le modèle retenu (régression linéaire), ses scores sur le jeu de test et la justification du choix."),
]:
    im = Image(img_path)
    ratio = im.imageHeight / im.imageWidth
    im.drawWidth = 16 * cm
    im.drawHeight = 16 * cm * ratio
    story.append(im)
    story.append(Paragraph(caption, styles["Caption"]))

story.append(PageBreak())

# ------------------------------------------------------ RECOMMANDATIONS ----
story.append(Paragraph("Recommandations agronomiques", styles["H1"]))
story.append(ListFlowable([
    ListItem(Paragraph(
        "<b>Prioriser l'irrigation et la fertilisation.</b> C'est le levier le plus efficace "
        "identifié dans les données : le gain moyen observé (+2,70 t/ha entre le meilleur et "
        "le pire scénario) dépasse largement les écarts liés au choix de la culture ou au type "
        "de sol.", styles["BulletItem"])),
    ListItem(Paragraph(
        "<b>Utiliser l'outil de recommandation comme aide à la décision, pas comme verdict "
        "unique.</b> Sur les données actuelles, les 6 cultures affichent des rendements "
        "estimés très proches, souvent à quelques centièmes de tonne par hectare, alors que "
        "l'erreur moyenne du modèle est d'environ 0,50 t/ha. Le classement ne permet donc "
        "généralement pas d'affirmer qu'une culture donnera un meilleur rendement qu'une "
        "autre : l'application le signale explicitement lorsque les écarts sont plus faibles "
        "que cette erreur. Le choix de culture doit aussi intégrer d'autres critères non "
        "couverts ici (prix de vente, coûts de production, rotation des cultures, débouchés "
        "commerciaux).",
        styles["BulletItem"])),
    ListItem(Paragraph(
        "<b>Tenir compte de la pluviométrie.</b> C'est le facteur naturel le plus lié au "
        "rendement (environ +0,5 t/ha par tranche de 100 mm de pluie par an). Sur les "
        "parcelles peu arrosées, l'irrigation est d'autant plus utile pour compenser.",
        styles["BulletItem"])),
    ListItem(Paragraph(
        "<b>Enrichir les données à l'avenir</b> avec des quantités réelles de pesticides et "
        "d'engrais par parcelle (plutôt que des indicateurs oui/non), ainsi que la localisation "
        "géographique précise, pour affiner encore les recommandations.", styles["BulletItem"])),
], bulletType="bullet", start="•"))

story.append(Paragraph("Limites à connaître", styles["H2"]))
story.append(ListFlowable([
    ListItem(Paragraph(
        "Les données de pesticides mondiales sont des totaux nationaux (tous produits "
        "confondus), pas des quantités spécifiques à chaque culture.", styles["BulletItem"])),
    ListItem(Paragraph(
        "L'orge et le coton n'ayant pas de correspondance directe dans les données climatiques "
        "mondiales utilisées, une moyenne globale leur a été appliquée par défaut.", styles["BulletItem"])),
    ListItem(Paragraph(
        "Le modèle a été entraîné sur un échantillon de 200 000 parcelles simulées ; une "
        "validation sur des données réelles de terrain reste recommandée avant un déploiement "
        "à grande échelle.", styles["BulletItem"])),
    ListItem(Paragraph(
        "La mise à jour de l'application n'est pas encore entièrement automatique : le "
        "réentraînement du modèle se lance à la main, et l'interface en ligne est mise à jour "
        "sans attendre la réussite des tests automatiques (l'API, elle, les attend). Le détail "
        "figure dans la documentation technique du projet.", styles["BulletItem"])),
], bulletType="bullet", start="•"))

story.append(Spacer(1, 20))
story.append(Paragraph(
    "Pour toute question sur ce rapport ou sur l'application, je reste disponible.",
    styles["Body"],
))

doc = SimpleDocTemplate(
    "reports/rapport_metier.pdf",
    pagesize=A4,
    topMargin=2 * cm,
    bottomMargin=2 * cm,
    leftMargin=2 * cm,
    rightMargin=2 * cm,
    title="Rapport métier - Agritech Answers",
    author="Fatima Adda - Agritech Answers",
)
doc.build(story)
print("Rapport généré : reports/rapport_metier.pdf")
