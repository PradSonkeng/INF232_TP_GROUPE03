"""
data_generator.py
------------------
Genere le jeu de donnees du groupe pour le THEME B :
"Plateforme de mise en relation freelance/client".

Variables produites pour chaque freelance simule :
  - note_moyenne_client   : performance du freelance, mesuree comme la
                            note moyenne (sur 5) laissee par les clients
                            a l'issue des missions terminees. C'est un
                            indicateur de performance directement
                            interpretable par une fondatrice non
                            statisticienne.
  - tarif_horaire_fcfa    : mesure d'activite/tarification du
                            freelance : son tarif horaire affiche sur
                            la plateforme, en FCFA. Choisi car c'est
                            une information immediatement disponible
                            des l'inscription (contrairement au volume
                            de missions, qui ne serait connu qu'a
                            posteriori).
  - profil_commercial     : etiquette "premium" / "standard" attribuee
                            historiquement par l'equipe commerciale,
                            de maniere imparfaite (le sujet precise
                            "un peu au feeling, sans methode claire").

Conception du generateur
-------------------------
Pour que les questions 3 (classification non supervisee) et 4
(classification supervisee) aient un sens statistique, les donnees
sont construites en deux temps :

1) Trois "profils naturels" latents de freelances (Debutant, Confirme,
   Expert) sont definis, chacun avec sa propre moyenne de performance
   et de tarif, et une correlation positive entre les deux variables
   au sein de chaque profil (un freelance plus performant a tendance a
   pouvoir demander un tarif plus eleve). Ces profils ne sont JAMAIS
   communiques a l'analyse : ils ne servent qu'a generer des donnees
   realistes, et c'est a la classification non supervisee (Q3) de les
   retrouver.

2) L'etiquette commerciale "premium"/"standard" (Q4) est ensuite
   simulee comme une decision humaine imparfaite : elle depend
   majoritairement de la performance et du tarif, mais avec un bruit
   deliberement important, pour refleter le fait que le classement a
   ete fait "au feeling" et n'est pas parfaitement previsible a partir
   des deux mesures (ce qui est essentiel pour repondre honnetement a
   la question 4 sur la confiance a accorder a une prediction
   automatique).

Taille de l'echantillon : n = 300 freelances.
Justification : un ordre de grandeur de quelques centaines
d'individus est necessaire pour (a) estimer une correlation et une
regression de facon stable, (b) permettre a un algorithme de
clustering (K-Means) de distinguer plusieurs profils sans etre
domine par du bruit d'echantillonnage, et (c) reserver un jeu de
test d'une taille raisonnable (environ 60 individus avec un split
80/20) pour evaluer honnetement un modele de classification
supervisee.
"""

import numpy as np
import pandas as pd

from seed_generator import name_to_seed

N_FREELANCES = 300

# Definition des 3 profils latents (jamais fournis a l'analyse)
# (poids, moyenne_note, ecart_type_note, moyenne_tarif, ecart_type_tarif, correlation_intra_profil)
PROFILS_LATENTS = [
    {"nom": "Debutant",  "poids": 0.40, "mu_note": 3.0, "sd_note": 0.45, "mu_tarif": 2200, "sd_tarif": 500,  "rho": 0.55},
    {"nom": "Confirme",  "poids": 0.35, "mu_note": 3.9, "sd_note": 0.35, "mu_tarif": 4500, "sd_tarif": 900,  "rho": 0.60},
    {"nom": "Expert",    "poids": 0.25, "mu_note": 4.6, "sd_note": 0.25, "mu_tarif": 8500, "sd_tarif": 1600, "rho": 0.50},
]

NOTE_MIN, NOTE_MAX = 1.0, 5.0
TARIF_MIN, TARIF_MAX = 800, 15000


def _generate_correlated_pair(rng, mu_x, sd_x, mu_y, sd_y, rho, n):
    """Genere n paires (x, y) correlees via une loi normale bivariee."""
    cov = rho * sd_x * sd_y
    mean = [mu_x, mu_y]
    cov_matrix = [[sd_x ** 2, cov], [cov, sd_y ** 2]]
    return rng.multivariate_normal(mean, cov_matrix, size=n)


def generate_dataset(nom_chef_groupe: str, n: int = N_FREELANCES) -> pd.DataFrame:
    """
    Genere le jeu de donnees complet et deterministe du groupe,
    a partir du nom du chef de groupe.
    """
    seed = name_to_seed(nom_chef_groupe)
    rng = np.random.default_rng(seed)

    lignes = []
    profil_latent_assigne = []

    # Repartition du nombre d'individus par profil latent (proportionnelle aux poids)
    poids = np.array([p["poids"] for p in PROFILS_LATENTS])
    effectifs = (poids * n).round().astype(int)
    effectifs[-1] += n - effectifs.sum()  # ajustement d'arrondi

    for profil, effectif in zip(PROFILS_LATENTS, effectifs):
        paires = _generate_correlated_pair(
            rng, profil["mu_note"], profil["sd_note"],
            profil["mu_tarif"], profil["sd_tarif"], profil["rho"], effectif
        )
        lignes.append(paires)
        profil_latent_assigne += [profil["nom"]] * effectif

    donnees = np.vstack(lignes)
    note = np.clip(donnees[:, 0], NOTE_MIN, NOTE_MAX)
    tarif = np.clip(donnees[:, 1], TARIF_MIN, TARIF_MAX)

    # Melange aleatoire (pour ne pas laisser les profils groupes par ordre)
    ordre = rng.permutation(n)
    note = note[ordre]
    tarif = tarif[ordre]
    profil_latent_assigne = np.array(profil_latent_assigne)[ordre]

    # --- Etiquette commerciale "premium" / "standard" (Q4) ---
    # Score fonde principalement sur note + tarif (standardises), + bruit important
    note_z = (note - note.mean()) / note.std()
    tarif_z = (tarif - tarif.mean()) / tarif.std()
    bruit = rng.normal(0, 1.1, size=n)  # bruit volontairement fort : decision "au feeling"
    score_commercial = 0.9 * note_z + 0.7 * tarif_z + bruit
    seuil = np.quantile(score_commercial, 0.60)  # ~40% de "premium"
    profil_commercial = np.where(score_commercial >= seuil, "premium", "standard")

    df = pd.DataFrame({
        "id_freelance": [f"FL{str(i+1).zfill(4)}" for i in range(n)],
        "note_moyenne_client": note.round(2),
        "tarif_horaire_fcfa": tarif.round(0).astype(int),
        "profil_commercial": profil_commercial,
    })

    # Colonne cachee (verite terrain du profil latent), utile pour la
    # verification interne du groupe mais NON utilisee dans les analyses
    # presentees a la commanditaire (elle n'est jamais connue dans la vraie vie).
    df["_profil_latent_verification_interne"] = profil_latent_assigne

    return df, seed


if __name__ == "__main__":
    df, seed = generate_dataset("Ngueyé Maurice")
    print(f"Graine du groupe : {seed}")
    print(df.head(10).to_string(index=False))
    print(f"\nTaille du jeu de donnees : {len(df)} freelances")
    print(df["profil_commercial"].value_counts())
