"""
utils/heatmap.py
=================

Construction de la heatmap gaussienne de densité de regard.

Choix méthodologique important (à documenter dans le mémoire) :
-----------------------------------------------------------------
Calculer une matrice à la résolution native de l'image (3840 x 2160, soit
plus de 8 millions de cases) puis lui appliquer un filtre gaussien est
inutilement coûteux en mémoire et en temps de calcul pour une simple
visualisation, et doit de toute façon être recalculé à la demande
(bouton "Mettre à jour la heatmap", voir section 15 du cahier des
charges). On calcule donc la heatmap sur une grille RÉDUITE d'un facteur
HEATMAP_DOWNSCALE (identique en largeur et en hauteur), ce qui :
  - conserve exactement le même rapport largeur/hauteur que l'image
    d'origine (3840/HEATMAP_DOWNSCALE x 2160/HEATMAP_DOWNSCALE) : il n'y a
    donc AUCUNE déformation, seulement une résolution plus grossière ;
  - est ensuite affiché exactement à l'échelle de l'image d'origine en
    convertissant les indices de la grille réduite en coordonnées de
    l'image complète (voir x_coords / y_coords ci-dessous), affichage géré
    par utils.plotting.add_heatmap_trace.
Les coordonnées des échantillons ou des fixations utilisées pour CONSTRUIRE
la heatmap restent, elles, les coordonnées originales en 3840x2160 : seule
la matrice de sortie est en résolution réduite. La précision analytique
(détection de fixations, calcul du TTFF, etc.) n'est donc jamais affectée
par ce choix, qui ne concerne que le rendu de la heatmap.
"""

from __future__ import annotations

import numpy as np
import streamlit as st
from scipy.ndimage import gaussian_filter

from utils.data import IMAGE_HEIGHT, IMAGE_WIDTH

# Facteur de réduction de la grille de calcul de la heatmap.
# 3840/4 = 960 et 2160/4 = 540 : des entiers exacts, pratiques et
# largement suffisants visuellement pour une image affichée à l'écran.
HEATMAP_DOWNSCALE = 4


@st.cache_data(show_spinner="Calcul de la heatmap...")
def build_heatmap(
    xs: np.ndarray,
    ys: np.ndarray,
    weights: np.ndarray,
    sigma_px: float,
    downscale: int = HEATMAP_DOWNSCALE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construit la matrice de densité lissée servant de heatmap.

    Paramètres
    ----------
    xs, ys : coordonnées (en pixels, résolution originale 3840x2160) des
        points contribuant à la heatmap. Il peut s'agir d'échantillons
        bruts valides, ou des positions moyennes de fixations.
    weights : poids de chaque point. Pour des échantillons bruts, on
        utilise un poids de 1 pour chacun (chaque échantillon contribue
        également). Pour des fixations, on utilise leur durée en
        millisecondes, comme demandé (section 13) : une fixation longue
        contribue donc davantage à la carte qu'une fixation brève.
    sigma_px : écart-type de la gaussienne de lissage, exprimé en pixels
        de l'image ORIGINALE (converti automatiquement pour la grille
        réduite, voir sigma_scaled ci-dessous).
    downscale : facteur de réduction de la grille de calcul.

    Retourne
    --------
    (z, x_coords, y_coords) :
    - z : matrice 2D (hauteur_réduite x largeur_réduite) normalisée entre
      0 et 1 (0 = aucune densité, 1 = maximum observé sur ce fichier).
    - x_coords, y_coords : tableaux 1D donnant, pour chaque colonne / ligne
      de z, la coordonnée correspondante dans l'image ORIGINALE (3840x2160).
      Ils permettent de positionner correctement le trace Plotly go.Heatmap
      par-dessus l'image sans aucune déformation.
    """
    small_w = IMAGE_WIDTH // downscale
    small_h = IMAGE_HEIGHT // downscale
    matrix = np.zeros((small_h, small_w), dtype=np.float64)

    if len(xs) > 0:
        # Conversion des coordonnées originales en indices de la grille
        # réduite. np.clip garantit qu'un point situé exactement sur le
        # bord de l'image (ex : X = 3839.9) ne provoque pas une erreur
        # d'index hors limites après troncature.
        col_idx = np.clip((xs / downscale).astype(int), 0, small_w - 1)
        row_idx = np.clip((ys / downscale).astype(int), 0, small_h - 1)

        # Accumulation des poids dans chaque case de la grille.
        # np.add.at gère correctement le cas où plusieurs points tombent
        # dans la même case (contrairement à une simple affectation).
        np.add.at(matrix, (row_idx, col_idx), weights)

    # Le sigma est exprimé par l'utilisateur en pixels de l'image
    # d'origine ; on le convertit à l'échelle de la grille réduite pour que
    # le lissage visuel corresponde bien à la valeur affichée dans l'UI.
    sigma_scaled = max(sigma_px / downscale, 1e-6)
    smoothed = gaussian_filter(matrix, sigma=sigma_scaled)

    # Normalisation entre 0 et 1 pour un affichage cohérent quel que soit
    # le nombre total de points ou la valeur du sigma choisi.
    max_val = smoothed.max()
    if max_val > 0:
        smoothed = smoothed / max_val

    # Coordonnées, dans le repère de l'image originale, du CENTRE de
    # chaque case de la grille réduite.
    x_coords = (np.arange(small_w) + 0.5) * downscale
    y_coords = (np.arange(small_h) + 0.5) * downscale

    return smoothed, x_coords, y_coords
