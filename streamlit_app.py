from __future__ import annotations
import numpy as np
import streamlit as st


from utils.data import (
    EYE_MODE_BINOCULAR,
    EYE_MODE_BOTH,
    EYE_MODE_LEFT,
    EYE_MODE_RIGHT,
    VALID_IMAGE_CODES,
    compute_binocular_point,
    compute_recording_summary,
    compute_validity,
    extract_image_code,
    get_gaze_series,
    load_experimental_image,
    parse_eyetracking_file,
)
from utils.fixations import compute_ttff_global, detect_fixations
from utils.heatmap import build_heatmap
from utils.plotting import (
    LEFT_EYE_COLOR,
    RIGHT_EYE_COLOR,
    SCANPATH_NUMBERING_WARNING_THRESHOLD,
    add_fixation_trace,
    add_heatmap_trace,
    add_scanpath_trace,
    create_base_figure,
)

# ---------------------------------------------------------------------------
# CONSTANTES DE L'INTERFACE (valeurs INITIALES uniquement)
# ---------------------------------------------------------------------------
# Ces valeurs par défaut ne constituent PAS des seuils validés
# scientifiquement : elles servent uniquement de point de départ pratique
# pour les widgets de l'interface. Elles sont volontairement regroupées ici
# pour pouvoir être changées facilement en un seul endroit.
DEFAULT_DISPERSION_THRESHOLD_PX = 50
DEFAULT_MIN_DURATION_MS = 100
DEFAULT_HEATMAP_SIGMA_PX = 60
DEFAULT_HEATMAP_OPACITY = 0.6

st.set_page_config(page_title="Analyse eye-tracking", layout="wide")
st.title("👁️ Data Analysis Eye-tracking")

# ---------------------------------------------------------------------------
# 1. CHARGEMENT DU FICHIER
# ---------------------------------------------------------------------------
uploaded_file = st.sidebar.file_uploader("Fichier de données (.txt)", type=["txt"])

if uploaded_file is None:
    st.info("Chargez un fichier de données (.txt) dans la barre latérale pour commencer.")
    st.stop()

file_bytes = uploaded_file.getvalue()

try:
    raw_df = parse_eyetracking_file(file_bytes, uploaded_file.name)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

df = compute_validity(raw_df)
df = compute_binocular_point(df)

# ---------------------------------------------------------------------------
# 2. IDENTIFICATION ET CHARGEMENT DE L'IMAGE EXPÉRIMENTALE
# ---------------------------------------------------------------------------
image_code = extract_image_code(uploaded_file.name)

if image_code is None or image_code not in VALID_IMAGE_CODES:
    st.error(
        f"Impossible de déterminer une image expérimentale valide à partir du "
        f"nom de fichier '{uploaded_file.name}'. Le nom doit suivre le format "
        f"DATA_<CODE>_<horodatage>.txt, avec CODE parmi : {VALID_IMAGE_CODES}."
    )
    st.stop()

image_array, image_error = load_experimental_image(image_code, display_downscale=4)
if image_error is not None:
    st.error(image_error)
    st.stop()

# Si l'on change de fichier, une heatmap calculée pour le fichier précédent
# n'a plus de sens : on la réinitialise automatiquement.
if st.session_state.get("heatmap_source_file") != uploaded_file.name:
    st.session_state["heatmap_result"] = None
    st.session_state["heatmap_source_file"] = uploaded_file.name

# ---------------------------------------------------------------------------
# 3. INDICATEURS GÉNÉRAUX SUR L'ENREGISTREMENT (au-dessus de la visualisation)
# ---------------------------------------------------------------------------
summary = compute_recording_summary(df)

st.subheader(f"Fichier : {uploaded_file.name}")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Image détectée", image_code)
c2.metric(
    "Réponse utilisateur",
    "N/A" if summary["user_response"] is None else f"{summary['user_response']:g}",
)
c3.metric("Durée totale", f"{summary['total_duration_ms']:.0f} ms")
c4.metric("Nombre d'échantillons", f"{summary['n_samples']}")
c5.metric(
    "Échantillons valides (G / D)",
    f"{summary['pct_valid_left']:.0f}% / {summary['pct_valid_right']:.0f}%",
)

st.divider()

# ---------------------------------------------------------------------------
# 4. SIDEBAR — PARAMÈTRES DU SCANPATH
# ---------------------------------------------------------------------------
st.sidebar.header("🧭 Scanpath")

with st.sidebar.expander("Affichage du scanpath", expanded=True):
    show_scanpath = st.checkbox("Afficher le scanpath", value=True)
    scanpath_eye_mode = st.radio(
        "Œil affiché",
        [EYE_MODE_LEFT, EYE_MODE_RIGHT, EYE_MODE_BOTH],
        index=2,
    )
    show_points = st.checkbox("Afficher les points", value=True)
    show_lines = st.checkbox("Afficher les lignes reliant les points", value=True)
    show_numbers = st.checkbox(
        "Numéroter les points",
        value=False,
        help=(
            "Désactivé par défaut : afficher un numéro sur chacun des "
            "milliers d'échantillons d'un fichier ralentit fortement "
            "l'affichage et rend la figure illisible. Combinez plutôt cette "
            "option avec le slider temporel ci-dessous, sur un extrait "
            "court du parcours."
        ),
    )
    if show_numbers and summary["n_samples"] > SCANPATH_NUMBERING_WARNING_THRESHOLD:
        st.caption(
            f"⚠️ {summary['n_samples']} échantillons : la numérotation complète "
            "peut ralentir l'affichage. Réduisez la plage temporelle ci-dessous "
            "si besoin."
        )

with st.sidebar.expander("Exploration temporelle", expanded=False):
    time_min = float(df["Time_ms"].min())
    time_max = float(df["Time_ms"].max())
    time_cursor = st.slider(
        "Temps affiché (ms)",
        min_value=time_min,
        max_value=time_max,
        value=time_max,  # par défaut : parcours complet affiché
        help="Seuls les échantillons dont le temps est inférieur ou égal à cette valeur sont affichés.",
    )

# ---------------------------------------------------------------------------
# 5. SIDEBAR — PARAMÈTRES DES FIXATIONS
# ---------------------------------------------------------------------------
st.sidebar.header("🎯 Fixations")

with st.sidebar.expander("Détection des fixations (I-DT)", expanded=True):
    st.caption(
        "Choix méthodologique V1 : la détection de fixations nécessite une "
        "position de regard UNIQUE à chaque instant (cf. section 6 du cahier "
        "des charges). L'œil utilisé pour ce calcul est donc sélectionné ici, "
        "indépendamment de l'œil affiché pour le scanpath ci-dessus."
    )
    fixation_eye_mode = st.selectbox(
        "Œil utilisé pour la détection",
        [EYE_MODE_BINOCULAR, EYE_MODE_LEFT, EYE_MODE_RIGHT],
        index=0,
    )
    dispersion_threshold_px = st.number_input(
        "Seuil de dispersion (px)",
        min_value=1,
        value=DEFAULT_DISPERSION_THRESHOLD_PX,
        step=5,
        help="Valeur initiale fournie à titre pratique, pas un seuil validé scientifiquement.",
    )
    min_duration_ms = st.number_input(
        "Durée minimale (ms)",
        min_value=1,
        value=DEFAULT_MIN_DURATION_MS,
        step=10,
        help="Valeur initiale fournie à titre pratique, pas un seuil validé scientifiquement.",
    )
    show_fixations = st.checkbox("Afficher les fixations", value=True)

# ---------------------------------------------------------------------------
# 6. CALCUL DES FIXATIONS
# ---------------------------------------------------------------------------
# Recalculé automatiquement à chaque changement de paramètre ou de fichier
# (mise en cache gérée par st.cache_data dans utils/fixations.py).
fixation_times, fixation_xs, fixation_ys = get_gaze_series(df, fixation_eye_mode)

if len(fixation_times) == 0:
    st.warning(
        f"Aucune coordonnée valide pour '{fixation_eye_mode}' : impossible de "
        "détecter des fixations avec cette source de regard."
    )
    fixations_df = detect_fixations(
        np.array([]), np.array([]), np.array([]), float(dispersion_threshold_px), float(min_duration_ms)
    )
else:
    fixations_df = detect_fixations(
        fixation_times, fixation_xs, fixation_ys, float(dispersion_threshold_px), float(min_duration_ms)
    )
    if fixations_df.empty:
        st.warning(
            "Aucune fixation détectée avec ces paramètres. Essayez d'augmenter "
            "le seuil de dispersion ou de réduire la durée minimale."
        )

ttff_global = compute_ttff_global(fixations_df)

# ---------------------------------------------------------------------------
# 7. INDICATEURS LIÉS AUX FIXATIONS
# ---------------------------------------------------------------------------
f1, f2, f3, f4 = st.columns(4)
f1.metric("Nombre de fixations", f"{len(fixations_df)}")
f2.metric(
    "Durée moyenne des fixations",
    "N/A" if fixations_df.empty else f"{fixations_df['duration'].mean():.0f} ms",
)
f3.metric(
    "Durée médiane des fixations",
    "N/A" if fixations_df.empty else f"{fixations_df['duration'].median():.0f} ms",
)
f4.metric("TTFF global", "N/A" if ttff_global is None else f"{ttff_global:.0f} ms")

if not fixations_df.empty:
    with st.expander("Détail des fixations (tableau)"):
        st.dataframe(fixations_df, width='stretch', hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# 8. SIDEBAR — PARAMÈTRES DE LA HEATMAP
# ---------------------------------------------------------------------------
st.sidebar.header("🔥 Heatmap")

show_heatmap = st.sidebar.checkbox("Afficher la heatmap", value=False)

with st.sidebar.form("heatmap_form"):
    st.caption(
        "Le calcul (potentiellement coûteux) n'est déclenché qu'au clic sur "
        "le bouton ci-dessous, pas à chaque interaction avec les autres "
        "widgets de l'application."
    )
    heatmap_eye_mode = st.radio(
        "Œil source",
        [EYE_MODE_BINOCULAR, EYE_MODE_LEFT, EYE_MODE_RIGHT],
        index=0,
    )
    heatmap_data_type = st.radio(
        "Données utilisées",
        ["Échantillons bruts", "Fixations pondérées par durée"],
        index=0,
    )
    heatmap_sigma_px = st.slider(
        "Sigma de la gaussienne (px)", min_value=5, max_value=400, value=DEFAULT_HEATMAP_SIGMA_PX
    )
    heatmap_opacity = st.slider(
        "Opacité de la heatmap", min_value=0.0, max_value=1.0, value=DEFAULT_HEATMAP_OPACITY
    )
    heatmap_submit = st.form_submit_button("Mettre à jour la heatmap")

if heatmap_submit:
    heat_times, heat_xs, heat_ys = get_gaze_series(df, heatmap_eye_mode)

    if len(heat_times) == 0:
        st.warning(f"Aucune coordonnée valide pour '{heatmap_eye_mode}' : heatmap non calculée.")
        st.session_state["heatmap_result"] = None
    elif heatmap_data_type == "Échantillons bruts":
        # Chaque échantillon valide contribue également à la carte (poids = 1).
        weights = np.ones(len(heat_xs))
        z, x_coords, y_coords = build_heatmap(heat_xs, heat_ys, weights, float(heatmap_sigma_px))
        st.session_state["heatmap_result"] = (z, x_coords, y_coords, heatmap_opacity)
    else:
        # Fixations pondérées par leur durée : on réutilise les paramètres de
        # fixation actuels (seuil de dispersion / durée minimale), appliqués
        # à la source de regard choisie ici pour la heatmap (qui peut
        # différer de l'œil choisi pour les indicateurs de fixation ci-dessus).
        heat_fixations = detect_fixations(
            heat_times, heat_xs, heat_ys, float(dispersion_threshold_px), float(min_duration_ms)
        )
        if heat_fixations.empty:
            st.warning(
                "Aucune fixation détectée pour cette source avec les paramètres "
                "de fixation actuels : heatmap non calculée."
            )
            st.session_state["heatmap_result"] = None
        else:
            hx = heat_fixations["mean_x"].to_numpy()
            hy = heat_fixations["mean_y"].to_numpy()
            weights = heat_fixations["duration"].to_numpy()
            z, x_coords, y_coords = build_heatmap(hx, hy, weights, float(heatmap_sigma_px))
            st.session_state["heatmap_result"] = (z, x_coords, y_coords, heatmap_opacity)

if show_heatmap and st.session_state.get("heatmap_result") is None:
    st.info(
        "Réglez les paramètres de la heatmap dans la barre latérale puis "
        "cliquez sur 'Mettre à jour la heatmap'."
    )

# ---------------------------------------------------------------------------
# 9. CONSTRUCTION ET AFFICHAGE DE LA FIGURE PRINCIPALE
# ---------------------------------------------------------------------------
fig = create_base_figure(image_array)

# Couche heatmap (en dessous du scanpath et des fixations).
if show_heatmap and st.session_state.get("heatmap_result") is not None:
    z, x_coords, y_coords, opacity = st.session_state["heatmap_result"]
    add_heatmap_trace(fig, z, x_coords, y_coords, opacity)

# Couche scanpath brut.
if show_scanpath:
    if scanpath_eye_mode == EYE_MODE_BOTH:
        eyes_to_plot = [(EYE_MODE_LEFT, LEFT_EYE_COLOR), (EYE_MODE_RIGHT, RIGHT_EYE_COLOR)]
    elif scanpath_eye_mode == EYE_MODE_LEFT:
        eyes_to_plot = [(EYE_MODE_LEFT, LEFT_EYE_COLOR)]
    else:
        eyes_to_plot = [(EYE_MODE_RIGHT, RIGHT_EYE_COLOR)]

    for eye_mode, color in eyes_to_plot:
        times, xs, ys = get_gaze_series(df, eye_mode)
        mask = times <= time_cursor
        if not np.any(mask):
            st.warning(f"Aucun échantillon valide pour '{eye_mode}' jusqu'à {time_cursor:.0f} ms.")
            continue
        add_scanpath_trace(
            fig,
            xs[mask],
            ys[mask],
            color=color,
            name=eye_mode,
            show_points=show_points,
            show_lines=show_lines,
            show_numbers=show_numbers,
        )

# Couche fixations (au-dessus de tout le reste).
if show_fixations:
    add_fixation_trace(fig, fixations_df)

st.plotly_chart(fig, width='stretch')



