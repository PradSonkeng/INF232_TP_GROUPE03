"""
analysis.py
-----------
Regroupe les quatre blocs d'analyse statistique mobilises pour
repondre aux quatre questions du Theme B :

  Q1 -> analyse univariee   (statistique_univariee)
  Q2 -> analyse bivariee    (statistique_bivariee)
  Q3 -> classification non supervisee (clustering_non_supervise)
  Q4 -> classification supervisee     (classification_supervisee)

Chaque fonction retourne un dictionnaire de resultats numeriques
directement exploitables par l'application Streamlit (affichage et
export), accompagne des objets necessaires pour tracer les graphiques.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Q1 - Statistique univariee : distribution de la performance (note client)
# ---------------------------------------------------------------------------
def statistique_univariee(df: pd.DataFrame, colonne: str = "note_moyenne_client") -> dict:
    x = df[colonne].to_numpy()

    q1, mediane, q3 = np.percentile(x, [25, 50, 75])
    iqr = q3 - q1
    borne_basse = q1 - 1.5 * iqr
    borne_haute = q3 + 1.5 * iqr
    outliers = df[(x < borne_basse) | (x > borne_haute)]

    resultats = {
        "n": len(x),
        "moyenne": float(np.mean(x)),
        "ecart_type": float(np.std(x, ddof=1)),
        "mediane": float(mediane),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "asymetrie_skewness": float(stats.skew(x)),
        "borne_basse_outlier": float(borne_basse),
        "borne_haute_outlier": float(borne_haute),
        "nb_outliers": int(len(outliers)),
        "outliers_ids": outliers["id_freelance"].tolist() if "id_freelance" in outliers.columns else [],
        "valeurs": x,
    }
    return resultats


# ---------------------------------------------------------------------------
# Q2 - Statistique bivariee : lien performance <-> tarif, regression lineaire
# ---------------------------------------------------------------------------
def statistique_bivariee(df: pd.DataFrame,
                          col_x: str = "tarif_horaire_fcfa",
                          col_y: str = "note_moyenne_client") -> dict:
    x = df[col_x].to_numpy().reshape(-1, 1)
    y = df[col_y].to_numpy()

    r_pearson, p_value = stats.pearsonr(x.flatten(), y)

    modele = LinearRegression()
    modele.fit(x, y)
    y_pred = modele.predict(x)
    residus = y - y_pred
    r2 = modele.score(x, y)
    erreur_type_residus = float(np.std(residus, ddof=2))

    return {
        "r_pearson": float(r_pearson),
        "p_value": float(p_value),
        "pente": float(modele.coef_[0]),
        "ordonnee_origine": float(modele.intercept_),
        "r2": float(r2),
        "erreur_type_residus": erreur_type_residus,
        "x": x.flatten(),
        "y": y,
        "y_pred": y_pred,
        "modele": modele,
    }


def predire_note_depuis_tarif(modele: LinearRegression, tarif: float) -> float:
    return float(modele.predict(np.array([[tarif]]))[0])


# ---------------------------------------------------------------------------
# Q3 - Classification non supervisee : profils naturels de freelances (K-Means)
# ---------------------------------------------------------------------------
def clustering_non_supervise(df: pd.DataFrame,
                              colonnes=("note_moyenne_client", "tarif_horaire_fcfa"),
                              k_min: int = 2, k_max: int = 6,
                              k_choisi: int = None,
                              seed: int = 0) -> dict:
    X = df[list(colonnes)].to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inerties = []
    silhouettes = []
    ks = list(range(k_min, k_max + 1))
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(X_scaled)
        inerties.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    if k_choisi is None:
        k_choisi = ks[int(np.argmax(silhouettes))]

    km_final = KMeans(n_clusters=k_choisi, n_init=10, random_state=seed)
    labels_final = km_final.fit_predict(X_scaled)
    centres_originaux = scaler.inverse_transform(km_final.cluster_centers_)

    df_result = df.copy()
    df_result["cluster"] = labels_final

    descriptions = []
    for c in range(k_choisi):
        sous_groupe = df_result[df_result["cluster"] == c]
        descriptions.append({
            "cluster": c,
            "effectif": len(sous_groupe),
            "note_moyenne": float(sous_groupe["note_moyenne_client"].mean()),
            "tarif_moyen": float(sous_groupe["tarif_horaire_fcfa"].mean()),
        })

    return {
        "ks": ks,
        "inerties": inerties,
        "silhouettes": silhouettes,
        "k_choisi": k_choisi,
        "labels": labels_final,
        "centres": centres_originaux,
        "descriptions": descriptions,
        "df_avec_clusters": df_result,
    }


# ---------------------------------------------------------------------------
# Q4 - Classification supervisee : prediction du profil commercial
# ---------------------------------------------------------------------------
def classification_supervisee(df: pd.DataFrame,
                                colonnes=("note_moyenne_client", "tarif_horaire_fcfa"),
                                cible: str = "profil_commercial",
                                test_size: float = 0.2,
                                seed: int = 0) -> dict:
    X = df[list(colonnes)].to_numpy()
    y = (df[cible] == "premium").astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    modele = LogisticRegression()
    modele.fit(X_train_s, y_train)
    y_pred = modele.predict(X_test_s)
    y_proba = modele.predict_proba(X_test_s)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    rappel = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    return {
        "modele": modele,
        "scaler": scaler,
        "matrice_confusion": cm,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "rappel": float(rappel),
        "f1": float(f1),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "taux_premium_reel": float(y.mean()),
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def predire_profil_commercial(resultats_clf: dict, note: float, tarif: float) -> dict:
    X = resultats_clf["scaler"].transform(np.array([[note, tarif]]))
    proba_premium = resultats_clf["modele"].predict_proba(X)[0, 1]
    prediction = "premium" if proba_premium >= 0.5 else "standard"
    return {"prediction": prediction, "probabilite_premium": float(proba_premium)}
