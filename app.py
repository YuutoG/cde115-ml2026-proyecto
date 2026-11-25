"""Demostración de investigación; no es un dispositivo de diagnóstico."""

import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from proyecto.data import LABELS
from proyecto.imaging import read_image
from proyecto.models import display_background, load_models, overlay_map, predict_image

ROOT = Path(__file__).resolve().parent
LABEL_NAMES = {
    "pneumonia": "Neumonía",
    "pleural_effusion": "Derrame pleural",
    "tuberculosis": "Tuberculosis",
    "edema": "Edema",
}
FAMILY_NAMES = {"imagenet": "ImageNet", "xrv": "TorchXRayVision"}


@st.cache_resource(max_entries=1, scope="session", show_spinner=False)
def get_models(directory, device_name, artifact_signature):
    return load_models(directory, device_name)


def artifact_signature(directory):
    paths = [directory / "manifest.json", *sorted(directory.glob("*.pt"))]
    return tuple((path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in paths)


def model_directories():
    base = ROOT / "runs" / "experimento_01" / "models"
    paths = {family: base / family for family in FAMILY_NAMES}
    legacy = os.environ.get("XRAY_MODEL_DIR")
    if legacy:
        directory = Path(legacy)
        with (directory / "manifest.json").open(encoding="utf-8") as file:
            family = json.load(file).get("family", "xrv")
        if family not in paths:
            raise ValueError(f"Familia desconocida en {directory}: {family}")
        paths[family] = directory
    for family in paths:
        override = os.environ.get(f"XRAY_{family.upper()}_MODEL_DIR")
        if override:
            paths[family] = Path(override)
    return paths


def result_tables(probabilities, ensemble, manifest):
    thresholds = manifest["thresholds"]["ensemble"]
    table = pd.DataFrame({
        "Patología": [LABEL_NAMES[label] for label in LABELS],
        "Ensamble (%)": [100 * value for value in ensemble],
        "Umbral sens85 (%)": [None if value is None else 100 * value for value in thresholds],
        "Comparación": [
            "No disponible" if threshold is None else
            ("Igual o superior" if score >= threshold else "Inferior")
            for score, threshold in zip(ensemble, thresholds, strict=True)
        ],
    })
    order = table.sort_values("Ensamble (%)", ascending=False, kind="stable").index
    members = pd.DataFrame({
        "Patología": [LABEL_NAMES[label] for label in LABELS],
        **{name: [100 * value for value in values] for name, values in probabilities.items()},
    })
    return table.loc[order].reset_index(drop=True), members.loc[order].reset_index(drop=True)


def show_family(family, directory, image, image_hash, background, device):
    st.caption(f"Dispositivo: {device}")
    run = st.button(f"🔬 Analizar con {FAMILY_NAMES[family]}", key=f"run_{family}",
                    disabled=image is None, type="primary")
    key = f"result_{family}"
    label_key = f"label_{family}"
    if image is None:
        st.session_state.pop(key, None)
        st.session_state.pop(label_key, None)
        st.info("Carga una imagen y pulsa el botón para ejecutar esta familia.")
        return
    try:
        signature = artifact_signature(directory)
    except OSError as error:
        st.session_state.pop(key, None)
        st.error(f"No se pudieron cargar los modelos desde {directory}: {error}")
        return
    result = st.session_state.get(key)
    if result and (result["image_hash"] != image_hash or result["signature"] != signature):
        st.session_state.pop(key)
        st.session_state.pop(label_key, None)
        result = None
    if run:
        st.session_state.pop(key, None)
        try:
            with st.spinner(f"Ejecutando {FAMILY_NAMES[family]} y generando mapas..."):
                models, manifest = get_models(str(directory), device, signature)
                if manifest.get("family", "xrv") != family:
                    raise ValueError(f"El manifiesto no corresponde a la familia {family}")
                probabilities, ensemble, maps, fused = predict_image(
                    models, manifest, image, device=device
                )
            label = LABELS[int(ensemble.argmax())]
            st.session_state[label_key] = label
            result = {"image_hash": image_hash, "signature": signature, "label": label,
                      "manifest": manifest, "probabilities": probabilities, "ensemble": ensemble,
                      "maps": maps, "fused": fused}
            st.session_state[key] = result
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            st.error(f"No se pudo analizar la imagen con {FAMILY_NAMES[family]}: {error}")
            return
    if result is None:
        st.info("Pulsa el botón para obtener resultados con esta familia.")
        return

    label = st.selectbox("🩻 Patología para los mapas Grad-CAM", LABELS,
                         format_func=lambda value: LABEL_NAMES[value], key=label_key)
    if label != result["label"]:
        try:
            with st.spinner(f"Actualizando mapas para {LABEL_NAMES[label]}..."):
                models, manifest = get_models(str(directory), device, signature)
                if manifest.get("family", "xrv") != family:
                    raise ValueError(f"El manifiesto no corresponde a la familia {family}")
                probabilities, ensemble, maps, fused = predict_image(
                    models, manifest, image, LABELS.index(label), device=device
                )
            result = {"image_hash": image_hash, "signature": signature, "label": label,
                      "manifest": manifest, "probabilities": probabilities, "ensemble": ensemble,
                      "maps": maps, "fused": fused}
            st.session_state[key] = result
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            st.session_state.pop(key, None)
            st.error(f"No se pudieron actualizar los mapas de {FAMILY_NAMES[family]}: {error}")
            return

    manifest = result["manifest"]
    ranking, members = result_tables(result["probabilities"], result["ensemble"], manifest)
    scores_column, members_column = st.columns(2)
    with scores_column:
        st.markdown("#### 📊 Puntuaciones del ensamble")
        st.caption("Orden descendente, no urgencia ni diagnóstico. Comparación sin redondear.")
        st.dataframe(ranking, hide_index=True,
                     column_config={
                         "Ensamble (%)": st.column_config.NumberColumn(format="%.1f %%"),
                         "Umbral sens85 (%)": st.column_config.NumberColumn(format="%.1f %%"),
                     })
    with members_column:
        st.markdown("#### 🧮 Modelos individuales")
        st.caption("Salidas calibradas internamente; no son probabilidades clínicas verificadas.")
        st.dataframe(members, hide_index=True, column_config={
            name: st.column_config.NumberColumn(format="%.1f %%") for name in result["probabilities"]
        })
        st.caption("Pesos: " + " · ".join(
            f"{name}: {weight:.1%}" for name, weight in
            zip(result["probabilities"], manifest["ensemble_weights"], strict=True)
        ))
    st.markdown(f"#### 🩻 Mapas exploratorios · {LABEL_NAMES[label]}")
    st.caption("Grad-CAM destaca regiones asociadas a la salida del modelo. "
               "No es segmentación, localización confirmada, causalidad ni evidencia diagnóstica. "
               "La fusión es una media heurística.")
    panels = [*result["maps"].items(), ("Fusión heurística", result["fused"])]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 5), squeeze=False)
    for ax, (name, cam) in zip(axes[0], panels, strict=True):
        if cam is None:
            ax.imshow(background)
            ax.text(0.5, 0.5, "Mapa no disponible", transform=ax.transAxes, ha="center",
                    color="white", bbox={"facecolor": "black", "alpha": 0.7})
        else:
            ax.imshow(overlay_map(background, cam))
        ax.set_title(name)
        ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)


@st.dialog("📘 Leyenda de resultados", width="large")
def show_legend():
    st.markdown("""
**Probabilidad calibrada:** cada modelo transforma sus puntuaciones mediante temperaturas ajustadas
por patología en un conjunto interno de calibración. El porcentaje es una salida del modelo, no una
probabilidad clínica comprobada para esta persona. La calibración puede cambiar entre poblaciones.

**Ensamble:** media ponderada de las probabilidades calibradas de los modelos de una familia.
Las patologías no son excluyentes, sus porcentajes no tienen que sumar 100 % y no son directamente
comparables como si fueran probabilidades clínicas.

**Umbral sens85:** punto fijado en calibración para buscar sensibilidad de al menos 85 % y maximizar
especificidad entre los puntos elegibles. La comparación usa el umbral propio del ensamble para
cada patología; “igual o superior” no significa presencia de enfermedad e “inferior” no la descarta.
El 85 % no está garantizado en otras imágenes; un umbral ausente se muestra como no disponible.

**Modelos individuales:** cada columna es la salida calibrada de un miembro, antes de promediarla;
las diferencias entre modelos indican variación de sus salidas, no certeza ni validación clínica.
""")


st.set_page_config(page_title="Radiografías | Demostración", page_icon="🩻", layout="wide")
st.markdown("""
<style>
    .block-container {max-width: 1300px; padding-top: 3.75rem; padding-bottom: 2rem;}
    h1, h2, h3 {letter-spacing: -0.02em;}
    div[data-testid="stTabs"] button[role="tab"] {font-weight: 600;}
    .demo-hero {padding: 1.15rem 1.6rem; border-radius: 14px;
        background: linear-gradient(120deg, #123550, #126b75); color: #fff;
        box-shadow: 0 8px 22px rgba(13, 61, 75, .15);}
    .demo-hero .eyebrow {color: #b6e4de; font-size: .72rem; font-weight: 700;
        letter-spacing: .12em; text-transform: uppercase;}
    .demo-hero h1 {color: #fff; margin: .3rem 0; padding: 0; font-size: 1.8rem;}
    .demo-hero p {color: #e3f2f1; margin: 0; line-height: 1.4;}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="demo-hero">
    <div class="eyebrow">🩻 Investigación · Prototipo demostrativo</div>
    <h1>Análisis exploratorio de radiografías</h1>
    <p>🔬 Dos familias de modelos y mapas exploratorios para investigación.</p>
</div>
""", unsafe_allow_html=True)
st.error("Prototipo no validado clínicamente. No usar para diagnóstico, triaje ni decisiones clínicas.")
if st.button("📘 Ver leyenda de resultados", key="legend"):
    show_legend()

try:
    directories = model_directories()
except (OSError, ValueError, TypeError) as error:
    st.error(f"No se pudo leer la configuración XRAY_MODEL_DIR: {error}")
    st.stop()

device = "cuda" if torch.cuda.is_available() else "cpu"
image = image_hash = background = None
upload_column, preview_column = st.columns([1.5, 1], vertical_alignment="center")
with upload_column:
    uploaded = st.file_uploader("🩻 Radiografía PNG monocromática", type=["png"])
    st.caption("Máximo: 30 MB y 30 millones de píxeles. Procesada en memoria; "
               "no se guarda intencionadamente.")
    st.caption("PNG de más de 8 bits: se usa 65535 como máximo; no interprete la salida "
               "si el rango efectivo es distinto.")
    if uploaded is not None:
        try:
            if uploaded.size > 30 * 1024 * 1024:
                raise ValueError("La imagen supera el límite de 30 MB")
            image_hash = hashlib.sha256(uploaded.getvalue()).hexdigest()
            image = read_image(uploaded)
            background = display_background(image)
            if image.dtype.itemsize > 1:
                st.warning("Imagen de más de 8 bits: se usará 65535 como máximo. "
                           "No interprete la salida si el rango efectivo es distinto.")
        except (OSError, ValueError, RuntimeError) as error:
            for family in FAMILY_NAMES:
                st.session_state.pop(f"result_{family}", None)
            st.error(f"No se pudo procesar la radiografía: {error}")
            st.stop()
with preview_column:
    if background is not None:
        st.image(background, caption="Vista con contraste ajustado · solo demostración", width=220)
    else:
        st.caption("📷 La vista previa aparecerá aquí al cargar la imagen.")

imagenet_tab, xrv_tab = st.tabs(["ImageNet", "TorchXRayVision"])
with imagenet_tab:
    show_family("imagenet", directories["imagenet"], image, image_hash, background, device)
with xrv_tab:
    show_family("xrv", directories["xrv"], image, image_hash, background, device)
