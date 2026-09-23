import json
import os

import cv2
import numpy as np
import pandas as pd
import pydicom
import streamlit as st
import torch
from PIL import Image

# Librerías para Interpretabilidad
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn
from torchvision import models, transforms

# ==========================================
# 1. CONFIGURACIÓN INICIAL Y METADATOS
# ==========================================
st.set_page_config(
    page_title="Triaje IA - Radiografía Torácica", layout="wide", page_icon="🫁"
)

IMAGE_SIZE = (512, 512)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@st.cache_data
def cargar_metadatos_clinicos():
    with open("pesos_optuna.json", "r") as f:
        pesos = json.load(f)

    df = pd.read_csv("umbrales_optimos.csv")
    patologias = df["Patologia"].tolist()
    umbrales = df["Umbral_Optimo"].values

    return pesos, patologias, umbrales


PESOS_OPTUNA, PATOLOGIAS, UMBRALES = cargar_metadatos_clinicos()
NUM_CLASSES = len(PATOLOGIAS)


# ==========================================
# 2. CARGA DE MODELOS (COMITÉ DE EXPERTOS)
# ==========================================
@st.cache_resource
def instanciar_modelo(nombre):
    if nombre == "resnet":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        capa_cam = [model.layer4[-1]]
    elif nombre == "densenet":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
        capa_cam = [model.features[-1]]
    elif nombre == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
        capa_cam = [model.features[-1]]
    return model, capa_cam


@st.cache_resource
def cargar_comite():
    comite = {}
    for nombre in ["resnet", "densenet", "efficientnet"]:
        ruta = f"./models/{nombre}_best_model.pth"
        if os.path.exists(ruta):
            modelo, capa_cam = instanciar_modelo(nombre)

            state_dict_crudo = torch.load(ruta, map_location=device, weights_only=True)
            state_dict_limpio = {
                k.replace("_orig_mod.", ""): v for k, v in state_dict_crudo.items()
            }

            modelo.load_state_dict(state_dict_limpio)
            modelo.to(device)
            modelo.eval()
            comite[nombre] = {"modelo": modelo, "capa_cam": capa_cam}
    return comite


COMITE = cargar_comite()


# ==========================================
# 3. PREPROCESAMIENTO CLÍNICO (DICOM Y CLAHE)
# ==========================================
def procesar_dicom(archivo):
    dicom = pydicom.dcmread(archivo)
    pixel_array = dicom.pixel_array.astype(float)

    if (
        "PhotometricInterpretation" in dicom
        and dicom.PhotometricInterpretation == "MONOCHROME1"
    ):
        pixel_array = np.max(pixel_array) - pixel_array

    pixel_array = pixel_array - np.min(pixel_array)
    if np.max(pixel_array) > 0:
        pixel_array = (pixel_array / np.max(pixel_array)) * 255.0

    pixel_array = np.uint8(pixel_array)
    return Image.fromarray(pixel_array).convert("RGB")


def aplicar_clahe(imagen_pil):
    img_cv = np.array(imagen_pil.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_cv)
    img_clahe_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(img_clahe_rgb), img_clahe_rgb


# ==========================================
# 4. INTERFAZ DE USUARIO (UI) Y MOTOR DE TRIAJE
# ==========================================
# Estado de sesión para congelar memoria RAM y gestionar renderizado progresivo
if "ultimo_archivo" not in st.session_state:
    st.session_state.ultimo_archivo = None
    st.session_state.img_cruda = None
    st.session_state.img_clahe = None
    st.session_state.img_cv = None
    st.session_state.analisis_completado = False
    st.session_state.hallazgos_ordenados = []
    st.session_state.probabilidades_individuales = {}
    st.session_state.mapas_calculados = {}  # Memoria caché para los Grad-CAM

st.title("🫁 Sistema de Triaje Inteligente Torácico")
st.markdown("**Prototipo Funcional de Grado Clínico (Fase 3)**")
st.markdown(
    "Sistema soportado por un *Weighted Ensemble* (EfficientNet-B0, ResNet-50, DenseNet-121). "
    r"Optimizado con **Asymmetric Loss** y ecualización **CLAHE** para alta sensibilidad (Recall $\ge 80\%$). "
    "Soporte nativo para formatos estándar y **DICOM médico**."
)

with st.sidebar:
    st.header("⚙️ Motor Algorítmico")
    st.info(
        f"**Resolución de Inferencia:** {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}\n\n"
        f"**Modelos Habilitados:** {len(COMITE)}\n\n"
        f"**Procesamiento:** CLAHE Adaptativo"
    )

    st.subheader("Umbrales Dinámicos")
    for pat, umbral in zip(PATOLOGIAS, UMBRALES):
        st.caption(f"{pat}: **{umbral:.2f}**")

    st.warning(
        "⚠️ Uso exclusivo para investigación y triaje. Los datos se procesan localmente."
    )

archivo_subido = st.file_uploader(
    "Sube una radiografía torácica frontal (Formatos admitidos: DICOM, JPG, PNG)",
    type=["jpg", "jpeg", "png", "dcm", "dicom"],
)

if archivo_subido is not None:
    # --- SISTEMA DE CONGELACIÓN EN MEMORIA RAM ---
    if st.session_state.ultimo_archivo != archivo_subido.name:
        st.session_state.ultimo_archivo = archivo_subido.name
        nombre_archivo = archivo_subido.name.lower()

        try:
            if nombre_archivo.endswith((".dcm", ".dicom")):
                archivo_subido.seek(0)
                imagen_cruda_pil = procesar_dicom(archivo_subido)
            else:
                imagen_cruda_pil = Image.open(archivo_subido).convert("RGB")
        except Exception as e:
            st.error(f"Error al procesar el archivo. Detalles: {e}")
            st.stop()

        imagen_clahe_pil, imagen_clahe_cv = aplicar_clahe(imagen_cruda_pil)

        st.session_state.img_cruda = imagen_cruda_pil
        st.session_state.img_clahe = imagen_clahe_pil
        st.session_state.img_cv = imagen_clahe_cv

        # Reset de la lógica de inferencia
        st.session_state.analisis_completado = False
        st.session_state.hallazgos_ordenados = []
        st.session_state.probabilidades_individuales = {}
        st.session_state.mapas_calculados = {}

    imagen_cruda_pil = st.session_state.img_cruda
    imagen_clahe_pil = st.session_state.img_clahe
    imagen_clahe_cv = st.session_state.img_cv

    st.subheader("Inspección Radiológica")
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image(imagen_cruda_pil, caption="Radiografía Original", width="stretch")
    with col_img2:
        st.image(
            imagen_clahe_pil,
            caption="Filtro de Ecualización (CLAHE)",
            width="stretch",
        )

    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    imagen_tensor = transform(imagen_clahe_pil).unsqueeze(0).to(device)
    imagen_float = cv2.resize(imagen_clahe_cv / 255.0, IMAGE_SIZE)

    # --- FASE 1: PREDICCIÓN INSTANTÁNEA ---
    if st.button(
        "🔬 Iniciar Triaje Algorítmico", type="primary", use_container_width=True
    ):
        with st.spinner("Evaluando anatomía con el Comité de IA..."):
            probabilidades_totales = np.zeros(NUM_CLASSES)
            probabilidades_individuales = {}

            # Inferencia pura rápida (ResNet50, DenseNet121, EfficientNet-B0)
            for nombre, datos in COMITE.items():
                modelo = datos["modelo"]
                peso = PESOS_OPTUNA[nombre]

                with torch.no_grad():
                    with torch.amp.autocast(
                        device_type="cuda" if "cuda" in str(device) else "cpu"
                    ):
                        logits = modelo(imagen_tensor)
                    probs = torch.sigmoid(logits.float())[0].cpu().numpy()

                    probabilidades_individuales[nombre] = probs
                    probabilidades_totales += probs * peso

            hallazgos = []
            for i, patologia in enumerate(PATOLOGIAS):
                prob = probabilidades_totales[i]
                umbral = UMBRALES[i]
                if prob >= umbral:
                    hallazgos.append((patologia, prob, i))

            st.session_state.hallazgos_ordenados = sorted(
                hallazgos, key=lambda x: x[1], reverse=True
            )
            st.session_state.probabilidades_individuales = probabilidades_individuales
            st.session_state.analisis_completado = True

    # --- FASE 2: RENDERIZADO PROGRESIVO DE GRAD-CAM ---
    if st.session_state.analisis_completado:
        st.markdown("---")
        st.subheader("📋 Reporte Diagnóstico Consolidado")

        hallazgos_ordenados = st.session_state.hallazgos_ordenados
        probabilidades_individuales = st.session_state.probabilidades_individuales

        if len(hallazgos_ordenados) == 0:
            st.success(
                "✅ **Sin hallazgos patológicos significativos detectados.** Todas las predicciones están por debajo del umbral clínico óptimo."
            )
        else:
            st.markdown("Generando mapas anatómicos en tiempo real...")

            contenedores_ui = {}

            # 1. Dibujar la estructura visual de inmediato (Acordeones vacíos con spinner)
            for patologia, prob_final, idx_pat in hallazgos_ordenados:
                tipo_alerta = "🚨" if prob_final > 0.8 else "⚠️"
                titulo = f"{tipo_alerta} Análisis de {patologia} (Confianza: {prob_final * 100:.1f}%)"

                with st.expander(titulo, expanded=True):
                    st.progress(
                        float(prob_final),
                        text=f"Umbral Clínico de Detección: {UMBRALES[idx_pat]:.2f}",
                    )
                    # Creamos un marcador de posición vacío dentro del expander
                    contenedores_ui[patologia] = st.empty()

                    # Si aún no está calculado en memoria, mostramos un mensaje de espera
                    if patologia not in st.session_state.mapas_calculados:
                        contenedores_ui[patologia].info(
                            f"⏳ Extrayendo topología de {patologia}..."
                        )

            # 2. Ejecutar los cálculos pesados y rellenar los marcadores progresivamente
            for patologia, prob_final, idx_pat in hallazgos_ordenados:
                # Calcular SÓLO si no existe en la RAM
                if patologia not in st.session_state.mapas_calculados:
                    mapas_base = {}
                    for nombre, datos in COMITE.items():
                        cam = GradCAM(
                            model=datos["modelo"], target_layers=datos["capa_cam"]
                        )
                        targets = [ClassifierOutputTarget(idx_pat)]
                        mapas_base[nombre] = cam(
                            input_tensor=imagen_tensor, targets=targets
                        )[0, :]

                    st.session_state.mapas_calculados[patologia] = mapas_base

                # Rescatar de la memoria RAM
                mapas_base = st.session_state.mapas_calculados[patologia]

                # Rellenar el marcador de posición con el contenido final
                with contenedores_ui[patologia].container():
                    tab_consenso, tab_lider, tab_comite = st.tabs(
                        [
                            "🤝 Consenso Ponderado (Ensemble)",
                            "🏆 Modelo Líder",
                            "🔬 Desglose del Comité",
                        ]
                    )

                    with tab_consenso:
                        cam_consenso = np.zeros(
                            (IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=np.float32
                        )
                        for nombre in COMITE.keys():
                            cam_consenso += mapas_base[nombre] * PESOS_OPTUNA[nombre]

                        if np.max(cam_consenso) > 0:
                            cam_consenso = (cam_consenso - np.min(cam_consenso)) / (
                                np.max(cam_consenso) - np.min(cam_consenso)
                            )

                        vis_consenso = show_cam_on_image(
                            imagen_float, cam_consenso, use_rgb=True
                        )

                        c1, c2 = st.columns([2, 1])
                        with c1:
                            st.image(
                                vis_consenso,
                                caption=f"Mapa Unificado - {patologia}",
                                width="stretch",
                            )
                        with c2:
                            st.info(
                                "**Vista para Médico de Triaje**\n\nFusión matemática de los 3 modelos, priorizando las zonas de acuerdo absoluto."
                            )

                    with tab_lider:
                        lider_nombre = max(
                            COMITE.keys(),
                            key=lambda k: probabilidades_individuales[k][idx_pat],
                        )
                        confianza_lider = probabilidades_individuales[lider_nombre][
                            idx_pat
                        ]

                        vis_lider = show_cam_on_image(
                            imagen_float, mapas_base[lider_nombre], use_rgb=True
                        )

                        c1, c2 = st.columns([2, 1])
                        with c1:
                            st.image(
                                vis_lider,
                                caption=f"Mapa Nativo de {lider_nombre.upper()} - {patologia}",
                                width="stretch",
                            )
                        with c2:
                            st.warning(
                                f"**Vista de Auditoría**\n\nEl modelo **{lider_nombre.upper()}** empujó este diagnóstico con una certeza cruda del **{confianza_lider * 100:.1f}%**."
                            )

                    with tab_comite:
                        st.markdown(
                            "**Aportes individuales de cada arquitectura al diagnóstico final:**"
                        )
                        cols = st.columns(3)

                        for col, nombre in zip(cols, COMITE.keys()):
                            vis_individual = show_cam_on_image(
                                imagen_float, mapas_base[nombre], use_rgb=True
                            )
                            confianza = probabilidades_individuales[nombre][idx_pat]
                            peso = PESOS_OPTUNA[nombre]

                            with col:
                                st.image(vis_individual, width="stretch")
                                st.caption(
                                    f"**{nombre.upper()}**\n\nCerteza cruda: {confianza * 100:.1f}%\nPeso en el ensamble: {peso * 100:.1f}%"
                                )
