"""
utils/plotting.py
==================

Construction de la figure Plotly : image expérimentale + couches
optionnelles (heatmap, scanpath, fixations).

Point essentiel sur l'orientation de l'axe Y (à ne jamais casser) :
--------------------------------------------------------------------
Dans les données de l'eye-tracker comme dans l'image, Y = 0 correspond au
HAUT de l'image, et Y augmente vers le BAS. C'est aussi la convention de
go.Image (ligne 0 du tableau numpy = haut de l'image).

Plotly, par défaut, oriente son axe Y vers le HAUT (comme un graphique
mathématique classique). Si on ne fait rien, l'image et les points de
regard apparaîtraient donc correctement superposés entre eux, mais
l'ensemble serait affiché "à l'envers" à l'écran par rapport à l'image
réelle.

La solution utilisée ici N'EST PAS de transformer les données (surtout pas
de calcul du type y = hauteur - y, qui inverserait le sens des données et
fausserait toute analyse ultérieure). La solution est uniquement
d'inverser le sens d'affichage de l'axe (yaxis.autorange="reversed") :
cela ne change AUCUNE valeur, seulement l'ordre dans lequel l'axe est
dessiné à l'écran. Toutes les coordonnées manipulées dans ce projet
(scanpath, fixations, heatmap) restent donc, du début à la fin, les
coordonnées brutes telles que fournies par l'eye-tracker.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils.data import IMAGE_HEIGHT, IMAGE_WIDTH

# Couleurs utilisées pour distinguer les couches (facilement modifiables).
LEFT_EYE_COLOR = "#1f77b4"     # bleu
RIGHT_EYE_COLOR = "#d62728"    # rouge
BINOCULAR_COLOR = "#2ca02c"    # vert
FIXATION_COLOR = "#ff7f0e"     # orange

# Taille des marqueurs de fixation (en pixels écran), en fonction de leur
# durée. La taille minimale garantit qu'une fixation très courte reste
# visible ; la taille maximale évite qu'une fixation très longue ne masque
# une grande partie de l'image.
FIXATION_MIN_MARKER_SIZE = 12
FIXATION_MAX_MARKER_SIZE = 40

# Nombre de points au-delà duquel on prévient l'utilisateur que la
# numérotation du scanpath peut ralentir l'affichage. Cette valeur est
# purement informative (voir add_scanpath_trace) : elle ne bloque jamais
# la fonctionnalité, conformément au cahier des charges.
SCANPATH_NUMBERING_WARNING_THRESHOLD = 500


def create_base_figure(image_array: np.ndarray) -> go.Figure:
    """
    Crée la figure Plotly de base : uniquement l'image expérimentale,
    avec un système d'axes qui garantit :
    - une correspondance pixel-à-pixel exacte avec les coordonnées de
      l'eye-tracker (aucune mise à l'échelle) ;
    - un rapport largeur/hauteur fixe, donc aucune déformation visuelle
      (xaxis.scaleanchor="y", scaleratio=1 : un pixel en X occupe toujours
      la même taille à l'écran qu'un pixel en Y) ;
    - une orientation verticale correcte (voir docstring du module).
    """
    fig = go.Figure()

    # go.Image attend un tableau (hauteur, largeur, canaux). Avec les
    # réglages par défaut (x0=0, y0=0, dx=1, dy=1), la colonne j du
    # tableau correspond exactement à la coordonnée X = j, et la ligne i
    # correspond exactement à la coordonnée Y = i : aucune conversion
    # supplémentaire n'est nécessaire.
    
    #fig.add_trace(go.Image(z=image_array, name="Image expérimentale", hoverinfo="skip"))

    fig.update_xaxes(
        range=[0, IMAGE_WIDTH],
        visible=False,
        constrain="domain",
    )
    fig.update_yaxes(
        range=[0, IMAGE_HEIGHT],
        visible=False,
        scaleanchor="x",
        scaleratio=1,
        autorange="reversed",  # affichage uniquement, voir docstring du module
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    return fig


def add_heatmap_trace(fig: go.Figure, z: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, opacity: float) -> None:
    """
    Ajoute la couche heatmap par-dessus l'image (mais sous le scanpath et
    les fixations, car les traces Plotly se superposent dans leur ordre
    d'ajout).

    z est déjà normalisée entre 0 et 1 par utils.heatmap.build_heatmap :
    zmin/zmax sont donc fixés explicitement ici pour garantir une échelle
    de couleur stable et cohérente.
    """
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=x_coords,
            y=y_coords,
            opacity=opacity,
            zmin=0,
            zmax=1,
            colorscale="Jet",
            showscale=True,
            colorbar=dict(title="Densité<br>(relative)", len=0.6),
            hoverinfo="skip",
            name="Heatmap",
            showlegend=False,
        )
    )


def add_scanpath_trace(
    fig: go.Figure,
    xs: np.ndarray,
    ys: np.ndarray,
    color: str,
    name: str,
    show_points: bool,
    show_lines: bool,
    show_numbers: bool,
) -> None:
    """
    Ajoute la trace du scanpath (parcours brut) pour UNE source de regard.

    Les points sont reliés dans leur ordre chronologique (xs/ys sont déjà
    triés par temps par utils.data.get_gaze_series) et numérotés dans ce
    même ordre si show_numbers est actif.

    Remarque de performance : afficher un texte (numéro) sur chacun des
    plusieurs milliers de points d'un fichier dégrade nettement la
    fluidité du graphique Plotly et rend la figure illisible (chiffres
    superposés). C'est pourquoi cette option est désactivée par défaut
    dans l'interface (voir streamlit_app.py) ; elle reste toutefois
    pleinement fonctionnelle si l'utilisateur souhaite l'activer, par
    exemple sur un extrait court obtenu via le slider temporel.
    """
    if len(xs) == 0:
        return

    mode_parts = []
    if show_lines:
        mode_parts.append("lines")
    if show_points:
        mode_parts.append("markers")
    if show_numbers:
        mode_parts.append("text")

    if not mode_parts:
        # L'utilisateur a décoché points, lignes ET numéros : on n'affiche
        # rien pour cette couche plutôt que d'imposer un rendu par défaut.
        return

    mode = "+".join(mode_parts)
    text = [str(i + 1) for i in range(len(xs))] if show_numbers else None

    fig.add_trace(
        go.Scattergl(
            x=xs,
            y=ys,
            mode=mode,
            name=name,
            line=dict(color=color, width=1.5),
            marker=dict(color=color, size=5),
            text=text,
            textposition="top center",
            textfont=dict(size=9, color=color),
            hovertemplate="X=%{x:.0f}, Y=%{y:.0f}<extra>%{fullData.name}</extra>",
        )
    )


def add_fixation_trace(fig: go.Figure, fixations_df: pd.DataFrame, name: str = "Fixations") -> None:
    """
    Ajoute la couche des fixations : un marqueur par fixation, positionné
    sur (mean_x, mean_y), numéroté dans l'ordre chronologique
    (fixation_id), avec une taille de marqueur proportionnelle à la durée
    de la fixation.
    """
    if fixations_df.empty:
        return

    durations = fixations_df["duration"].to_numpy(dtype=float)
    d_min, d_max = durations.min(), durations.max()
    if d_max > d_min:
        sizes = FIXATION_MIN_MARKER_SIZE + (durations - d_min) / (d_max - d_min) * (
            FIXATION_MAX_MARKER_SIZE - FIXATION_MIN_MARKER_SIZE
        )
    else:
        # Toutes les fixations ont la même durée (ou il n'y en a qu'une) :
        # on utilise une taille intermédiaire fixe pour éviter une
        # division par zéro.
        sizes = np.full_like(durations, (FIXATION_MIN_MARKER_SIZE + FIXATION_MAX_MARKER_SIZE) / 2)

    fig.add_trace(
        go.Scatter(
            x=fixations_df["mean_x"],
            y=fixations_df["mean_y"],
            mode="markers+text",
            name=name,
            marker=dict(
                size=sizes,
                color=FIXATION_COLOR,
                opacity=0.55,
                line=dict(color="black", width=1),
            ),
            text=fixations_df["fixation_id"].astype(str),
            textposition="middle center",
            textfont=dict(size=10, color="black"),
            customdata=fixations_df[["duration", "n_samples"]],
            hovertemplate=(
                "Fixation #%{text}<br>Durée=%{customdata[0]:.0f} ms"
                "<br>Échantillons=%{customdata[1]}<extra></extra>"
            ),
        )
    )
