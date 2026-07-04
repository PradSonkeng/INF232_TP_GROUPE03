# TP INF232 — Groupe 3 — Thème B (Plateforme freelance/client)

## Mode d'emploi (3 lignes)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Puis ouvrir l'adresse indiquée dans le terminal (généralement http://localhost:8501).

## Contenu du dossier

- `seed_generator.py` — transforme le nom du chef de groupe en graine déterministe.
- `data_generator.py` — génère le jeu de données (300 freelances) à partir de la graine.
- `analysis.py` — les 4 blocs d'analyse (univariée, bivariée, clustering, classification).
- `app.py` — application Streamlit interactive (onglets Q1 à Q4 + simulateurs + exports).
- `donnees_freelances_groupe03.csv` — extrait des données générées (pré-exporté).

## Fonctionnalités interactives

- Régénération des données avec un autre nom / une autre taille d'échantillon.
- Simulateurs en direct : estimer une note à partir d'un tarif (Q2), tester une
  prédiction de profil commercial pour un nouveau freelance (Q4).
- Choix manuel ou automatique du nombre de profils en clustering (Q3).
- Export individuel de chaque graphique (PNG) et du jeu de données (CSV).
- Export global : archive `.zip` regroupant données, graphiques et synthèse JSON
  des résultats chiffrés des 4 questions (onglet "Export global").

Graine du groupe (chef : Ngueyé Maurice) : **2881825947**
