"""
utils/fixations.py
===================

Détection de fixations oculaires par la méthode I-DT
(Identification by Dispersion Threshold).

Une seule fonction publique : detect_fixations(...).
Elle prend en entrée une série temporelle DÉJÀ FILTRÉE sur les échantillons
valides d'UNE SEULE source de regard (œil gauche, œil droit ou point
binoculaire — voir utils.data.get_gaze_series), triée par temps croissant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# DÉFINITION MATHÉMATIQUE RETENUE (à citer telle quelle dans le mémoire)
# ---------------------------------------------------------------------------
# Pour une fenêtre de points consécutifs, la DISPERSION est définie comme :
#
#       dispersion = (max(X) - min(X)) + (max(Y) - min(Y))
#
# Une fenêtre de points constitue une FIXATION si et seulement si :
#   (1) sa dispersion est INFÉRIEURE OU ÉGALE à un seuil de dispersion
#       "distance_threshold_px" (exprimé en pixels) ;
#   (2) sa durée (temps du dernier point - temps du premier point) est
#       SUPÉRIEURE OU ÉGALE à un seuil temporel "min_duration_ms"
#       (exprimé en millisecondes).
#
# ALGORITHME (I-DT classique, tel que décrit dans le cahier des charges) :
#   1. Partir du premier échantillon valide (index i).
#   2. Construire une fenêtre [i, j] contenant AU MINIMUM la durée
#      min_duration_ms.
#   3. Calculer la dispersion de cette fenêtre.
#      - Si elle dépasse déjà le seuil : aucune fixation ne peut commencer
#        en i ; on avance i d'un échantillon et on recommence.
#      - Sinon : on ÉTEND progressivement la fenêtre (en ajoutant les
#        échantillons suivants un par un) tant que la dispersion reste
#        sous le seuil.
#   4. Dès que l'ajout d'un nouvel échantillon ferait dépasser le seuil,
#      on ARRÊTE l'extension : la fenêtre précédente (avant cet ajout) est
#      enregistrée comme une fixation.
#   5. Pour cette fixation, on calcule : heure de début, heure de fin,
#      durée, X moyen, Y moyen, nombre d'échantillons.
#   6. On reprend l'algorithme à partir du premier échantillon SUIVANT la
#      fixation qui vient d'être enregistrée (pas de recouvrement entre
#      deux fixations consécutives).
#
# Remarque d'implémentation (sans impact sur la définition ci-dessus) :
# plutôt que de recalculer min(X), max(X), min(Y), max(Y) sur toute la
# fenêtre à chaque extension (coût O(n) à chaque étape, donc O(n²) au
# total), on met à jour ces 4 valeurs de façon incrémentale à chaque ajout
# d'échantillon. Le résultat mathématique est rigoureusement identique,
# mais le calcul reste fluide même sur des fichiers de plusieurs milliers
# de points.
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def detect_fixations(
    times: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    distance_threshold_px: float,
    min_duration_ms: float,
) -> pd.DataFrame:
    """
    Détecte des fixations par seuillage de dispersion (I-DT).

    Paramètres
    ----------
    times, xs, ys : tableaux numpy de même longueur, triés par temps
        croissant, ne contenant QUE des échantillons valides d'une seule
        source de regard (œil gauche, œil droit ou point binoculaire).
    distance_threshold_px : seuil de dispersion maximal (en pixels).
    min_duration_ms : durée minimale d'une fixation (en millisecondes).

    Retourne
    --------
    Un DataFrame avec une ligne par fixation détectée et les colonnes :
    fixation_id, start_time, end_time, duration, mean_x, mean_y, n_samples.
    Le DataFrame est vide (mais avec les bonnes colonnes) si aucune
    fixation n'est détectée ou si l'entrée est vide.

    Les données invalides ne sont jamais interprétées comme des
    déplacements oculaires puisque cette fonction ne reçoit en entrée que
    des échantillons déjà filtrés comme valides (voir
    utils.data.get_gaze_series) : un "trou" temporel entre deux échantillons
    valides ne compte donc jamais comme un déplacement dans le calcul de
    dispersion.
    """
    columns = ["fixation_id", "start_time", "end_time", "duration", "mean_x", "mean_y", "n_samples"]
    n = len(times)
    if n == 0:
        return pd.DataFrame(columns=columns)

    fixations = []
    i = 0

    while i < n:
        # Étape 2 : étendre j jusqu'à atteindre la durée minimale.
        j = i
        while j < n - 1 and (times[j] - times[i]) < min_duration_ms:
            j += 1

        # Pas assez de points restants pour atteindre la durée minimale :
        # on ne peut plus former de nouvelle fixation, on arrête.
        if (times[j] - times[i]) < min_duration_ms:
            break

        # Dispersion de la fenêtre minimale [i, j].
        min_x, max_x = xs[i : j + 1].min(), xs[i : j + 1].max()
        min_y, max_y = ys[i : j + 1].min(), ys[i : j + 1].max()
        dispersion = (max_x - min_x) + (max_y - min_y)

        if dispersion > distance_threshold_px:
            # Aucune fixation ne peut commencer en i : on avance d'un cran.
            i += 1
            continue

        # Étape 3-4 : extension progressive tant que le seuil est respecté.
        k = j
        while k + 1 < n:
            new_x, new_y = xs[k + 1], ys[k + 1]
            candidate_min_x = min(min_x, new_x)
            candidate_max_x = max(max_x, new_x)
            candidate_min_y = min(min_y, new_y)
            candidate_max_y = max(max_y, new_y)
            candidate_dispersion = (candidate_max_x - candidate_min_x) + (candidate_max_y - candidate_min_y)

            if candidate_dispersion <= distance_threshold_px:
                k += 1
                min_x, max_x, min_y, max_y = candidate_min_x, candidate_max_x, candidate_min_y, candidate_max_y
                dispersion = candidate_dispersion
            else:
                break

        # Étape 5 : enregistrement de la fixation [i, k].
        fixations.append(
            {
                "start_time": float(times[i]),
                "end_time": float(times[k]),
                "duration": float(times[k] - times[i]),
                "mean_x": float(xs[i : k + 1].mean()),
                "mean_y": float(ys[i : k + 1].mean()),
                "n_samples": int(k - i + 1),
            }
        )

        # Étape 6 : reprise après la fixation, sans recouvrement.
        i = k + 1

    result = pd.DataFrame(fixations, columns=[c for c in columns if c != "fixation_id"])
    if not result.empty:
        result.insert(0, "fixation_id", range(1, len(result) + 1))
    else:
        result = pd.DataFrame(columns=columns)

    return result


def compute_ttff_global(fixations_df: pd.DataFrame) -> float | None:
    """
    TTFF global (Time To First Fixation), tel que défini dans le cahier
    des charges :

        TTFF global = timestamp de début de la première fixation détectée,
        relativement à t = 0 (c'est-à-dire égal à "start_time" de la
        fixation n°1, le temps du fichier étant déjà exprimé depuis le
        début de l'enregistrement).

    Retourne None s'il n'existe aucune fixation (l'appelant est
    responsable d'afficher "N/A" dans ce cas).
    """
    if fixations_df.empty:
        return None
    return float(fixations_df.iloc[0]["start_time"])
