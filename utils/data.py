"""
utils/data.py
=============

Ce module regroupe tout ce qui concerne les DONNÉES BRUTES :
- lecture du fichier .txt exporté par l'eye-tracker ;
- identification de l'image expérimentale associée au fichier ;
- chargement de cette image depuis assets/images/ ;
- calcul de la validité de chaque échantillon (œil gauche / œil droit) ;
- calcul du "point binoculaire" (position de regard combinée) ;
- petites fonctions de résumé (durée, nombre d'échantillons, etc.).

Aucune fonction de ce fichier ne dessine quoi que ce soit : c'est le rôle
de utils/plotting.py. Cette séparation permet de tester / modifier le
traitement des données sans toucher à l'affichage, et inversement.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# CONSTANTES GÉNÉRALES DU PROJET
# ---------------------------------------------------------------------------
# Dimensions de l'image expérimentale (identiques pour les 6 images).
IMAGE_WIDTH = 3840
IMAGE_HEIGHT = 2160

# Les 6 codes d'image autorisés, tels qu'ils apparaissent dans le nom du
# fichier de données : DATA_<CODE>_<horodatage>.txt
VALID_IMAGE_CODES = ["REF1", "REF2", "COMP1", "COMP2", "COMP3", "TRIO"]

# Dossier contenant les 6 images de référence (toujours livrées avec le
# dépôt GitHub, jamais uploadées par l'utilisateur).
# On construit le chemin à partir de l'emplacement de ce fichier plutôt que
# du répertoire courant, ce qui rend l'application robuste quel que soit
# l'endroit depuis lequel Streamlit est lancé.
IMAGES_DIR = Path(__file__).resolve().parent.parent / "assets" / "images"

# Colonnes attendues dans le fichier de données, après renommage de
# "Time(ms)" en "Time_ms" (voir parse_eyetracking_file).
EXPECTED_COLUMNS = ["Time_ms", "Left_X", "Left_Y", "Right_X", "Right_Y", "User_Response"]


# ---------------------------------------------------------------------------
# IDENTIFICATION DU CODE IMAGE ET CHARGEMENT DE L'IMAGE
# ---------------------------------------------------------------------------
def extract_image_code(filename: str) -> str | None:
    """
    Extrait le code image à partir du nom de fichier.

    Format attendu : DATA_<CODE>_<n'importe quoi>.txt
    Exemple : "DATA_REF1_20250527_115716.txt" -> "REF1"

    Retourne None si le nom de fichier ne respecte pas ce format
    (la validité du code lui-même, c'est-à-dire son appartenance à la
    liste VALID_IMAGE_CODES, est vérifiée séparément par l'appelant).
    """
    match = re.match(r"^DATA_([A-Za-z0-9]+)_", filename)
    if match is None:
        return None
    return match.group(1)


def load_experimental_image(code: str) -> tuple[np.ndarray | None, str | None]:
    """
    Charge l'image expérimentale correspondant au code donné.

    Retourne un tuple (image_array, message_erreur).
    - Si tout va bien : (tableau numpy RGB, None)
    - En cas de problème : (None, "message d'erreur explicite")

    L'image est retournée sous forme de tableau numpy (hauteur, largeur, 3)
    car c'est le format attendu par le trace Plotly go.Image utilisé dans
    utils/plotting.py.
    """
    image_path = IMAGES_DIR / f"{code}.png"

    if not image_path.exists():
        return None, (
            f"L'image de référence pour le code '{code}' est introuvable "
            f"({image_path}). Vérifiez qu'elle a bien été placée dans "
            f"assets/images/ sous le nom '{code}.png'."
        )

    image = Image.open(image_path).convert("RGB")

    # Vérification de cohérence : on s'attend à une image 3840x2160.
    # On ne bloque pas l'application si ce n'est pas le cas (cela pourrait
    # arriver avec une image de test), mais on prévient l'utilisateur car
    # cela casserait la correspondance pixel-à-pixel avec les coordonnées
    # de regard.
    if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
        st.warning(
            f"L'image '{code}.png' fait {image.size[0]}x{image.size[1]} px "
            f"au lieu de {IMAGE_WIDTH}x{IMAGE_HEIGHT} px attendus. "
            "La correspondance entre les coordonnées de regard et les "
            "pixels de l'image risque d'être incorrecte."
        )

    return np.array(image), None


# ---------------------------------------------------------------------------
# LECTURE ET NETTOYAGE DU FICHIER DE DONNÉES
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def parse_eyetracking_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Lit et nettoie un fichier de données brut.

    Le fichier est au format suivant :
        Time(ms);Left_X;Left_Y;Right_X;Right_Y;User_Response
        4,57;1592;1268;1646;1249;3
        ...
    - séparateur de colonnes : ";"
    - séparateur décimal : "," (d'où le paramètre decimal="," ci-dessous)

    Cette fonction est mise en cache (st.cache_data) car le parsing d'un
    fichier de plusieurs milliers de lignes n'a pas besoin d'être refait à
    chaque interaction de l'utilisateur avec les widgets de la sidebar :
    tant que le fichier (ses octets) ne change pas, le résultat est réutilisé.

    Lève une ValueError avec un message clair en cas de problème
    (colonnes manquantes, fichier illisible, etc.) : c'est à l'appelant
    (streamlit_app.py) d'attraper cette exception et de l'afficher
    proprement avec st.error, sans faire planter l'application.
    """
    import io

    try:
        df = pd.read_csv(io.BytesIO(file_bytes), sep=";", decimal=",")
    except Exception as exc:  # noqa: BLE001 - on veut un message générique et sûr
        raise ValueError(
            f"Impossible de lire le fichier '{filename}' comme un fichier "
            f".txt délimité par ';'. Détail technique : {exc}"
        ) from exc

    # On renomme "Time(ms)" en "Time_ms" pour éviter les problèmes liés aux
    # parenthèses dans les noms de colonnes (accès par attribut, etc.).
    df = df.rename(columns={"Time(ms)": "Time_ms"})

    missing_columns = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(
            "Le fichier ne contient pas les colonnes attendues. "
            f"Colonnes manquantes : {missing_columns}. "
            f"Colonnes trouvées : {list(df.columns)}."
        )

    # Conversion explicite en valeurs numériques. errors="coerce" transforme
    # toute valeur illisible (texte, cellule vide, etc.) en NaN plutôt que
    # de faire planter le programme. Ces NaN seront ensuite traités comme
    # des coordonnées invalides par compute_validity, ce qui répond
    # naturellement à l'exigence "données numériques illisibles" du cahier
    # des charges (section 17) sans code supplémentaire.
    for col in EXPECTED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df.empty:
        raise ValueError("Le fichier ne contient aucune ligne de données exploitable.")

    return df


# ---------------------------------------------------------------------------
# VALIDITÉ DES ÉCHANTILLONS
# ---------------------------------------------------------------------------
def compute_validity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute deux colonnes booléennes 'Left_valid' et 'Right_valid'.

    Une coordonnée (X, Y) est valide si et seulement si :
        0 <= X < IMAGE_WIDTH   et   0 <= Y < IMAGE_HEIGHT

    Ce test unique gère naturellement TOUS les cas d'invalidité mentionnés
    dans le cahier des charges, sans traitement particulier :
    - la valeur -1 (utilisée par l'eye-tracker pour signaler une perte de
      suivi) échoue au test "X >= 0" ;
    - les coordonnées négatives échouent également à ce test ;
    - les coordonnées X > 3840 échouent au test "X < IMAGE_WIDTH" ;
    - les valeurs NaN (issues d'un nettoyage de données illisibles, voir
      parse_eyetracking_file) échouent automatiquement à toute comparaison
      en pandas/numpy, et sont donc correctement classées comme invalides.

    Important : les échantillons invalides ne sont JAMAIS modifiés ici
    (pas de clip, pas de remplacement par 0). On se contente de les
    marquer ; ce sont les fonctions de calcul en aval (fixations, heatmap,
    scanpath) qui doivent filtrer sur ces colonnes avant tout calcul.
    """
    df = df.copy()
    df["Left_valid"] = (
        (df["Left_X"] >= 0)
        & (df["Left_X"] < IMAGE_WIDTH)
        & (df["Left_Y"] >= 0)
        & (df["Left_Y"] < IMAGE_HEIGHT)
    )
    df["Right_valid"] = (
        (df["Right_X"] >= 0)
        & (df["Right_X"] < IMAGE_WIDTH)
        & (df["Right_Y"] >= 0)
        & (df["Right_Y"] < IMAGE_HEIGHT)
    )
    return df


def compute_binocular_point(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le "point binoculaire" (Bino_X, Bino_Y, Bino_valid).

    Logique demandée par le cahier des charges (isolée ici dans une seule
    fonction afin de pouvoir la faire évoluer facilement plus tard) :
    - si les DEUX yeux sont valides à un instant donné : on prend la
      moyenne des deux positions ;
    - si UN SEUL œil est valide : on utilise temporairement la position de
      cet œil ;
    - si AUCUN œil n'est valide : l'échantillon binoculaire est invalide
      (Bino_X, Bino_Y = NaN, Bino_valid = False).

    Doit être appelée après compute_validity (elle dépend des colonnes
    Left_valid / Right_valid).
    """
    df = df.copy()

    both_valid = df["Left_valid"] & df["Right_valid"]
    only_left = df["Left_valid"] & ~df["Right_valid"]
    only_right = df["Right_valid"] & ~df["Left_valid"]

    bino_x = pd.Series(np.nan, index=df.index, dtype=float)
    bino_y = pd.Series(np.nan, index=df.index, dtype=float)

    bino_x[both_valid] = (df.loc[both_valid, "Left_X"] + df.loc[both_valid, "Right_X"]) / 2
    bino_y[both_valid] = (df.loc[both_valid, "Left_Y"] + df.loc[both_valid, "Right_Y"]) / 2

    bino_x[only_left] = df.loc[only_left, "Left_X"]
    bino_y[only_left] = df.loc[only_left, "Left_Y"]

    bino_x[only_right] = df.loc[only_right, "Right_X"]
    bino_y[only_right] = df.loc[only_right, "Right_Y"]

    df["Bino_X"] = bino_x
    df["Bino_Y"] = bino_y
    df["Bino_valid"] = both_valid | only_left | only_right

    return df


# ---------------------------------------------------------------------------
# ACCÈS AUX SÉRIES DE REGARD PAR MODE (GAUCHE / DROIT / BINOCULAIRE)
# ---------------------------------------------------------------------------
# Libellés utilisés dans toute l'interface pour désigner les 3 modes.
EYE_MODE_LEFT = "Œil gauche"
EYE_MODE_RIGHT = "Œil droit"
EYE_MODE_BOTH = "Les deux yeux"       # utilisé uniquement pour l'AFFICHAGE du scanpath
EYE_MODE_BINOCULAR = "Binoculaire"    # utilisé pour les calculs à position unique


def get_gaze_series(df: pd.DataFrame, eye_mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retourne les échantillons VALIDES (temps, X, Y) pour un mode donné
    parmi EYE_MODE_LEFT, EYE_MODE_RIGHT, EYE_MODE_BINOCULAR.

    Cette fonction est le point d'entrée unique utilisé par la détection de
    fixations et par la heatmap pour obtenir "une seule position de regard
    à la fois" (section 6 du cahier des charges). Le mode EYE_MODE_BOTH
    n'a pas de sens ici (il n'existe que pour l'affichage superposé du
    scanpath) et lève une erreur explicite s'il est utilisé par erreur.

    Les tableaux retournés sont triés par ordre chronologique (le fichier
    source l'est déjà, mais on trie explicitement par robustesse).
    """
    if eye_mode == EYE_MODE_LEFT:
        valid_mask = df["Left_valid"]
        x_col, y_col = "Left_X", "Left_Y"
    elif eye_mode == EYE_MODE_RIGHT:
        valid_mask = df["Right_valid"]
        x_col, y_col = "Right_X", "Right_Y"
    elif eye_mode == EYE_MODE_BINOCULAR:
        valid_mask = df["Bino_valid"]
        x_col, y_col = "Bino_X", "Bino_Y"
    else:
        raise ValueError(
            f"get_gaze_series ne gère pas le mode '{eye_mode}' : ce mode ne "
            "correspond pas à une position de regard unique."
        )

    subset = df.loc[valid_mask, ["Time_ms", x_col, y_col]].sort_values("Time_ms")
    times = subset["Time_ms"].to_numpy(dtype=float)
    xs = subset[x_col].to_numpy(dtype=float)
    ys = subset[y_col].to_numpy(dtype=float)
    return times, xs, ys


# ---------------------------------------------------------------------------
# INDICATEURS SYNTHÉTIQUES SIMPLES
# ---------------------------------------------------------------------------
def compute_recording_summary(df: pd.DataFrame) -> dict:
    """
    Calcule les indicateurs globaux affichés en haut de l'application :
    durée totale, nombre d'échantillons, réponse utilisateur, et taux de
    validité par œil.
    """
    total_duration_ms = float(df["Time_ms"].max() - df["Time_ms"].min())
    n_samples = int(len(df))

    user_response_series = df["User_Response"].dropna()
    user_response = user_response_series.iloc[0] if not user_response_series.empty else None

    pct_valid_left = float(df["Left_valid"].mean() * 100)
    pct_valid_right = float(df["Right_valid"].mean() * 100)

    return {
        "total_duration_ms": total_duration_ms,
        "n_samples": n_samples,
        "user_response": user_response,
        "pct_valid_left": pct_valid_left,
        "pct_valid_right": pct_valid_right,
    }
