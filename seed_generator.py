"""
seed_generator.py
------------------
Transforme le nom complet du chef de groupe en une graine entiere
deterministe, utilisee ensuite pour initialiser le generateur de
nombres pseudo-aleatoires qui produit l'integralite du jeu de donnees.

Algorithme retenu (con u par le groupe) :
  1. Normalisation de la chaine : suppression des accents, mise en
     majuscules, suppression de tout caractere non alphabetique
     (espaces, tirets, apostrophes...).
  2. Concatenation prenom(s) + nom, sans separateur, comme demande
     dans l'annexe du sujet.
  3. Transformation en entier par un hachage polynomial de type
     "rolling hash" :
         seed = sum( (ord(c) - 64) * 31^i )  mod (2**32 - 1)
     Le facteur 31 est un multiplicateur premier classique pour ce
     type de hachage (utilise notamment dans la fonction hashCode de
     Java), il limite les collisions entre chaines proches.
  4. Le resultat est un entier positif reproductible, utilise comme
     seed d'un generateur numpy (PCG64 via numpy.random.default_rng).

Proprietes verifiees :
  - Determinisme : le meme nom produit toujours la meme graine.
  - Sensibilite : deux noms differents produisent, dans l'immense
    majorite des cas, des graines differentes (effet avalanche du
    hachage polynomial : permuter ou changer une seule lettre change
    fortement la valeur finale).
"""

import unicodedata

MODULO = 2**32 - 1
MULTIPLIER = 31


def _strip_accents(text: str) -> str:
    """Supprime les accents (e.g. 'e' -> 'e') via decomposition Unicode."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_name(nom_complet: str) -> str:
    """
    Nettoie le nom complet du chef de groupe :
    majuscules, sans accents, sans espaces ni ponctuation.
    """
    sans_accents = _strip_accents(nom_complet).upper()
    lettres_seules = "".join(ch for ch in sans_accents if ch.isalpha())
    return lettres_seules


def name_to_seed(nom_complet: str) -> int:
    """
    Convertit le nom complet du chef de groupe en une graine entiere
    deterministe et reproductible (voir docstring du module).
    """
    chaine = normalize_name(nom_complet)
    if not chaine:
        raise ValueError("Le nom fourni ne contient aucune lettre exploitable.")

    seed = 0
    for i, ch in enumerate(chaine):
        valeur_lettre = ord(ch) - 64  # 'A' -> 1, 'B' -> 2, ...
        seed = (seed + valeur_lettre * pow(MULTIPLIER, i, MODULO)) % MODULO

    # On evite une graine nulle (cas degenere improbable mais possible)
    return seed if seed != 0 else 1


if __name__ == "__main__":
    nom = "Ngueyé Maurice"
    print(f"Nom brut          : {nom}")
    print(f"Chaine normalisee : {normalize_name(nom)}")
    print(f"Graine obtenue    : {name_to_seed(nom)}")
