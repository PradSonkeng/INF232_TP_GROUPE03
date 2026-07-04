# -*- coding: utf-8 -*-
"""
app.py — Application interactive du TP INF232 (Groupe 3)
Theme B : Plateforme de mise en relation freelance/client

Lancement : streamlit run app.py
"""

import io
import json
import zipfile
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from seed_generator import name_to_seed, normalize_name
from data_generator import generate_dataset, N_FREELANCES
from analysis import (
    statistique_univariee, statistique_bivariee, predire_note_depuis_tarif,
    clustering_non_supervise, classification_supervisee, predire_profil_commercial,
)

st.set_page_config(page_title="TP INF232 - Groupe 3 - Theme B", layout="wide")

# ---------------------------------------------------------------------------
# Etat de session : donnees generees une seule fois par nom / taille choisis
# ---------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.seed = None
    st.session_state.nom_chef = None


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Barre laterale : generation des donnees
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Génération des données")
st.sidebar.markdown(
    "Les données sont générées de façon **déterministe** à partir du nom "
    "du chef de groupe (voir Annexe du sujet)."
)

nom_chef = st.sidebar.text_input("Nom complet du chef de groupe", value="Ngueyé Maurice")
n_freelances = st.sidebar.slider("Nombre de freelances à générer", min_value=100, max_value=1000,
                                  value=N_FREELANCES, step=50)

if st.sidebar.button("🎲 Générer / régénérer les données", type="primary"):
    df, seed = generate_dataset(nom_chef, n=n_freelances)
    st.session_state.df = df
    st.session_state.seed = seed
    st.session_state.nom_chef = nom_chef

if st.session_state.df is None:
    df, seed = generate_dataset(nom_chef, n=n_freelances)
    st.session_state.df = df
    st.session_state.seed = seed
    st.session_state.nom_chef = nom_chef

df = st.session_state.df
seed = st.session_state.seed

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Chaîne normalisée :** `{normalize_name(st.session_state.nom_chef)}`")
st.sidebar.markdown(f"**Graine du groupe :** `{seed}`")
st.sidebar.markdown(f"**Taille de l'échantillon :** {len(df)} freelances")

st.sidebar.markdown("---")
st.sidebar.subheader("📦 Export du jeu de données")
st.sidebar.download_button(
    "⬇️ Exporter les données (CSV)",
    data=df_to_csv_bytes(df.drop(columns=["_profil_latent_verification_interne"])),
    file_name="donnees_freelances_groupe03.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------------------
# En-tete
# ---------------------------------------------------------------------------
st.title("📊 TP INF232 — Groupe 3 — Thème B : Plateforme freelance/client")
st.caption("Application interactive : statistique univariée, bivariée, "
           "classification non supervisée et supervisée.")

onglets = st.tabs([
    "🗂️ Données brutes",
    "1️⃣ Q1 — Distribution de la performance",
    "2️⃣ Q2 — Lien performance / tarif",
    "3️⃣ Q3 — Profils naturels (non supervisé)",
    "4️⃣ Q4 — Prédiction du profil commercial",
    "📤 Export global",
])

# ===========================================================================
# ONGLET 0 : Donnees brutes
# ===========================================================================
with onglets[0]:
    st.subheader("Aperçu du jeu de données généré")
    st.dataframe(df.drop(columns=["_profil_latent_verification_interne"]), use_container_width=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Nombre de freelances", len(df))
    col2.metric("% premium", f"{(df['profil_commercial']=='premium').mean()*100:.1f} %")
    col3.metric("Graine du groupe", seed)

# ===========================================================================
# ONGLET 1 : Q1 - Statistique univariee
# ===========================================================================
with onglets[1]:
    st.subheader("Question 1 — Comment se répartit la performance des freelances ?")
    r1 = statistique_univariee(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Moyenne", f"{r1['moyenne']:.2f} / 5")
    c2.metric("Médiane", f"{r1['mediane']:.2f} / 5")
    c3.metric("Écart-type", f"{r1['ecart_type']:.2f}")
    c4.metric("Valeurs atypiques", r1["nb_outliers"])

    fig1, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(r1["valeurs"], bins=20, color="#4C72B0", edgecolor="white")
    axes[0].axvline(r1["moyenne"], color="red", linestyle="--", label=f"Moyenne = {r1['moyenne']:.2f}")
    axes[0].axvline(r1["mediane"], color="green", linestyle="--", label=f"Médiane = {r1['mediane']:.2f}")
    axes[0].set_title("Distribution de la note moyenne client")
    axes[0].set_xlabel("Note moyenne client (/5)")
    axes[0].set_ylabel("Nombre de freelances")
    axes[0].legend()

    axes[1].boxplot(r1["valeurs"], vert=True)
    axes[1].set_title("Boîte à moustaches (repérage des valeurs atypiques)")
    axes[1].set_ylabel("Note moyenne client (/5)")
    fig1.tight_layout()
    st.pyplot(fig1)

    st.download_button("⬇️ Exporter le graphique Q1 (PNG)", data=fig_to_png_bytes(fig1),
                        file_name="Q1_distribution_performance.png", mime="image/png")

    if r1["nb_outliers"] > 0:
        st.warning(f"{r1['nb_outliers']} freelance(s) atypique(s) détecté(s) : {r1['outliers_ids']}")
    else:
        st.info("Aucune valeur franchement atypique détectée selon le critère de l'IQR (1.5×IQR).")

    with st.expander("💬 Réponse rédigée pour la fondatrice (non statisticienne)"):
        st.markdown(f"""
En moyenne, nos freelances obtiennent une note client de **{r1['moyenne']:.2f} sur 5**
(la moitié d'entre eux ont une note en dessous de {r1['mediane']:.2f}, l'autre moitié au-dessus).
La grande majorité des notes se situent entre **{r1['q1']:.2f}** et **{r1['q3']:.2f}**.
{"Nous n'avons identifié aucun freelance dont la note sorte franchement du lot." if r1['nb_outliers']==0 else
f"Nous avons identifié {r1['nb_outliers']} freelance(s) dont la note sort nettement du lot (à surveiller ou à féliciter selon le sens de l'écart)."}

**Limite à garder en tête :** cette photographie dépend de l'échantillon observé ;
un échantillon plus petit ou biaisé (ex. seulement des freelances récents) donnerait une image différente.
""")

# ===========================================================================
# ONGLET 2 : Q2 - Statistique bivariee
# ===========================================================================
with onglets[2]:
    st.subheader("Question 2 — La performance et le tarif évoluent-ils ensemble ?")
    r2 = statistique_bivariee(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Corrélation de Pearson (r)", f"{r2['r_pearson']:.3f}")
    c2.metric("Coefficient de détermination (R²)", f"{r2['r2']:.3f}")
    c3.metric("p-value", f"{r2['p_value']:.2e}")

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.scatter(r2["x"], r2["y"], alpha=0.5, color="#4C72B0", label="Freelances observés")
    ordre = np.argsort(r2["x"])
    ax2.plot(r2["x"][ordre], r2["y_pred"][ordre], color="red", linewidth=2, label="Droite de régression")
    ax2.set_xlabel("Tarif horaire (FCFA)")
    ax2.set_ylabel("Note moyenne client (/5)")
    ax2.set_title("Relation entre tarif horaire et performance")
    ax2.legend()
    fig2.tight_layout()
    st.pyplot(fig2)

    st.download_button("⬇️ Exporter le graphique Q2 (PNG)", data=fig_to_png_bytes(fig2),
                        file_name="Q2_relation_tarif_performance.png", mime="image/png")

    st.markdown("#### 🔮 Simulateur : estimer la note à partir du tarif seul")
    tarif_simule = st.slider("Tarif horaire proposé (FCFA)", 800, 15000, 4000, step=100)
    note_estimee = predire_note_depuis_tarif(r2["modele"], tarif_simule)
    st.success(f"Note estimée pour un tarif de {tarif_simule} FCFA : **{note_estimee:.2f} / 5** "
               f"(± {r2['erreur_type_residus']:.2f} environ, incertitude typique du modèle)")

    tarif_min_donnees, tarif_max_donnees = df["tarif_horaire_fcfa"].min(), df["tarif_horaire_fcfa"].max()
    if tarif_simule < tarif_min_donnees or tarif_simule > tarif_max_donnees:
        st.error(f"⚠️ Ce tarif est en dehors de la plage observée dans les données "
                  f"({tarif_min_donnees:.0f} – {tarif_max_donnees:.0f} FCFA). "
                  "L'estimation devient une extrapolation non fiable.")

    with st.expander("💬 Réponse rédigée pour la fondatrice"):
        st.markdown(f"""
Oui, il existe un lien assez net entre le tarif horaire d'un freelance et sa performance
(corrélation de {r2['r_pearson']:.2f} sur une échelle de -1 à 1, statistiquement significative).
Le tarif seul permet d'expliquer environ **{r2['r2']*100:.0f}%** de la variation de la note client.

Il est donc envisageable d'estimer la performance probable à partir du seul tarif, **mais seulement
pour des tarifs proches de ceux déjà observés** (entre {tarif_min_donnees:.0f} et {tarif_max_donnees:.0f} FCFA).
Au-delà, l'estimation n'a plus de fondement dans les données et devient dangereuse à utiliser
commercialement (un freelance à tarif extrême n'a pas forcément le comportement extrapolé).
""")

# ===========================================================================
# ONGLET 3 : Q3 - Clustering non supervise
# ===========================================================================
with onglets[3]:
    st.subheader("Question 3 — Quels profils naturels se dégagent des données ?")

    k_manuel = st.checkbox("Choisir manuellement le nombre de profils (k)", value=False)
    k_valeur = None
    if k_manuel:
        k_valeur = st.slider("Nombre de profils (k)", 2, 6, 3)

    r3 = clustering_non_supervise(df, k_choisi=k_valeur, seed=seed % (2**31))

    fig3a, ax3a = plt.subplots(figsize=(6, 4))
    ax3a.plot(r3["ks"], r3["silhouettes"], marker="o", color="#55A868")
    ax3a.axvline(r3["k_choisi"], color="red", linestyle="--", label=f"k retenu = {r3['k_choisi']}")
    ax3a.set_xlabel("Nombre de profils (k)")
    ax3a.set_ylabel("Score de silhouette")
    ax3a.set_title("Qualité de séparation selon le nombre de profils")
    ax3a.legend()
    fig3a.tight_layout()

    fig3b, ax3b = plt.subplots(figsize=(7, 5))
    scatter = ax3b.scatter(df["tarif_horaire_fcfa"], df["note_moyenne_client"],
                            c=r3["labels"], cmap="viridis", alpha=0.6)
    ax3b.scatter(r3["centres"][:, 1], r3["centres"][:, 0], c="red", marker="X", s=200, label="Centres des profils")
    ax3b.set_xlabel("Tarif horaire (FCFA)")
    ax3b.set_ylabel("Note moyenne client (/5)")
    ax3b.set_title(f"Profils identifiés (k={r3['k_choisi']})")
    ax3b.legend()
    fig3b.tight_layout()

    col_g, col_d = st.columns(2)
    col_g.pyplot(fig3a)
    col_d.pyplot(fig3b)

    colb1, colb2 = st.columns(2)
    colb1.download_button("⬇️ Exporter le graphique silhouette (PNG)", data=fig_to_png_bytes(fig3a),
                           file_name="Q3_silhouette.png", mime="image/png")
    colb2.download_button("⬇️ Exporter le graphique des profils (PNG)", data=fig_to_png_bytes(fig3b),
                           file_name="Q3_profils.png", mime="image/png")

    st.markdown(f"**Nombre de profils retenu : {r3['k_choisi']}** (score de silhouette le plus élevé)")
    tableau_profils = pd.DataFrame(r3["descriptions"])
    tableau_profils.columns = ["Profil (n°)", "Effectif", "Note moyenne", "Tarif moyen (FCFA)"]
    st.dataframe(tableau_profils, use_container_width=True)

    st.download_button("⬇️ Exporter les données avec profils assignés (CSV)",
                        data=df_to_csv_bytes(r3["df_avec_clusters"].drop(columns=["_profil_latent_verification_interne"])),
                        file_name="Q3_donnees_avec_profils.csv", mime="text/csv")

    with st.expander("💬 Réponse rédigée pour la fondatrice"):
        lignes = "\n".join(
            f"- **Profil {d['cluster']}** ({d['effectif']} freelances) : note moyenne "
            f"{d['note_moyenne']:.2f}/5, tarif moyen {d['tarif_moyen']:.0f} FCFA."
            for d in r3["descriptions"]
        )
        st.markdown(f"""
Indépendamment de l'étiquette "premium/standard" déjà utilisée, vos données font ressortir
**{r3['k_choisi']} profils naturels** de freelances :

{lignes}

**Limite à garder en tête :** le nombre de profils n'est pas une vérité absolue — il dépend de la
méthode et du critère utilisés. D'autres découpages (k différent) restent statistiquement défendables ;
ce chiffre est une aide à la décision, pas une classification figée.
""")

# ===========================================================================
# ONGLET 4 : Q4 - Classification supervisee
# ===========================================================================
with onglets[4]:
    st.subheader("Question 4 — Peut-on prédire automatiquement le profil commercial ?")

    r4 = classification_supervisee(df, seed=seed % (2**31))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exactitude (accuracy)", f"{r4['accuracy']*100:.1f} %")
    c2.metric("Précision", f"{r4['precision']*100:.1f} %")
    c3.metric("Rappel", f"{r4['rappel']*100:.1f} %")
    c4.metric("Score F1", f"{r4['f1']*100:.1f} %")

    fig4, ax4 = plt.subplots(figsize=(5, 4))
    cm = r4["matrice_confusion"]
    im = ax4.imshow(cm, cmap="Blues")
    ax4.set_xticks([0, 1]); ax4.set_xticklabels(["standard", "premium"])
    ax4.set_yticks([0, 1]); ax4.set_yticklabels(["standard", "premium"])
    ax4.set_xlabel("Prédiction du modèle")
    ax4.set_ylabel("Étiquette réelle (commerciale)")
    ax4.set_title(f"Matrice de confusion (n test = {r4['n_test']})")
    for i in range(2):
        for j in range(2):
            ax4.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)
    fig4.tight_layout()
    st.pyplot(fig4)

    st.download_button("⬇️ Exporter la matrice de confusion (PNG)", data=fig_to_png_bytes(fig4),
                        file_name="Q4_matrice_confusion.png", mime="image/png")

    st.markdown("#### 🔮 Simulateur : tester la prédiction pour un nouveau freelance")
    col1, col2 = st.columns(2)
    note_test = col1.slider("Note moyenne client attendue (/5)", 1.0, 5.0, 4.0, step=0.1)
    tarif_test = col2.slider("Tarif horaire (FCFA)", 800, 15000, 4000, step=100)
    resultat_pred = predire_profil_commercial(r4, note_test, tarif_test)
    st.info(f"Profil prédit : **{resultat_pred['prediction'].upper()}** "
            f"(probabilité d'être premium : {resultat_pred['probabilite_premium']*100:.1f} %)")

    with st.expander("💬 Réponse rédigée pour la fondatrice"):
        st.markdown(f"""
Oui, un système automatique peut suggérer un profil ("premium" ou "standard") dès l'inscription,
uniquement à partir de la note et du tarif. Sur des freelances jamais vus par le modèle pendant son
apprentissage, il a raison dans **{r4['accuracy']*100:.0f}% des cas**.

Concrètement : sur {r4['n_test']} freelances-test, le modèle se trompe {r4['matrice_confusion'][0,1] + r4['matrice_confusion'][1,0]} fois
— tantôt en manquant un vrai profil premium (perte commerciale potentielle), tantôt en sur-classant un
profil standard en premium (avantages accordés à tort).

**Risque commercial :** {int((1-r4['accuracy'])*100)}% d'erreur reste non négligeable si le profil
premium donne accès à des avantages coûteux (mise en avant, réduction de commission...). Nous
recommandons d'utiliser cette prédiction comme **aide à la décision** pour l'équipe commerciale,
et non comme un classement automatique définitif, au moins tant que la fiabilité n'est pas éprouvée
sur un plus grand nombre de cas réels.
""")

# ===========================================================================
# ONGLET 5 : Export global
# ===========================================================================
with onglets[5]:
    st.subheader("📤 Export global — tous les résultats en un clic")
    st.markdown("Génère une archive `.zip` contenant : les données (CSV), les 5 graphiques (PNG) "
                "et une synthèse chiffrée (JSON) des 4 questions.")

    if st.button("📦 Préparer l'archive complète"):
        r1 = statistique_univariee(df)
        r2 = statistique_bivariee(df)
        r3 = clustering_non_supervise(df, seed=seed % (2**31))
        r4 = classification_supervisee(df, seed=seed % (2**31))

        synthese = {
            "groupe": "GROUPE03",
            "theme": "B - Plateforme freelance/client",
            "chef_de_groupe": st.session_state.nom_chef,
            "graine": int(seed),
            "date_export": datetime.now().isoformat(timespec="seconds"),
            "taille_echantillon": len(df),
            "Q1_statistique_univariee": {k: v for k, v in r1.items() if k not in ("valeurs",)},
            "Q2_statistique_bivariee": {k: v for k, v in r2.items() if k not in ("x", "y", "y_pred", "modele")},
            "Q3_clustering": {
                "k_choisi": r3["k_choisi"],
                "descriptions_profils": r3["descriptions"],
            },
            "Q4_classification": {k: v for k, v in r4.items()
                                   if k not in ("modele", "scaler", "y_test", "y_pred", "y_proba", "matrice_confusion")},
        }

        buf_zip = io.BytesIO()
        with zipfile.ZipFile(buf_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("donnees_freelances_groupe03.csv",
                        df.drop(columns=["_profil_latent_verification_interne"]).to_csv(index=False))
            zf.writestr("synthese_resultats_groupe03.json", json.dumps(synthese, indent=2, ensure_ascii=False))

            fig1, ax1 = plt.subplots(figsize=(7, 4))
            ax1.hist(r1["valeurs"], bins=20, color="#4C72B0", edgecolor="white")
            ax1.set_title("Q1 - Distribution de la performance")
            zf.writestr("Q1_distribution.png", fig_to_png_bytes(fig1))
            plt.close(fig1)

            fig2, ax2 = plt.subplots(figsize=(7, 5))
            ax2.scatter(r2["x"], r2["y"], alpha=0.5)
            ordre = np.argsort(r2["x"])
            ax2.plot(r2["x"][ordre], r2["y_pred"][ordre], color="red")
            ax2.set_title("Q2 - Relation tarif / performance")
            zf.writestr("Q2_relation.png", fig_to_png_bytes(fig2))
            plt.close(fig2)

            fig3, ax3 = plt.subplots(figsize=(7, 5))
            ax3.scatter(df["tarif_horaire_fcfa"], df["note_moyenne_client"], c=r3["labels"], cmap="viridis", alpha=0.6)
            ax3.set_title("Q3 - Profils identifiés")
            zf.writestr("Q3_profils.png", fig_to_png_bytes(fig3))
            plt.close(fig3)

            fig4, ax4 = plt.subplots(figsize=(5, 4))
            ax4.imshow(r4["matrice_confusion"], cmap="Blues")
            ax4.set_title("Q4 - Matrice de confusion")
            zf.writestr("Q4_confusion.png", fig_to_png_bytes(fig4))
            plt.close(fig4)

        buf_zip.seek(0)
        st.download_button("⬇️ Télécharger l'archive complète (.zip)", data=buf_zip,
                            file_name="TP_INF232_GROUPE03_resultats.zip", mime="application/zip")
        st.success("Archive prête ! Cliquez sur le bouton ci-dessus pour la télécharger.")
