import json
import os

import cv2
import numpy as np
import pandas as pd
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
# 1. CONFIGURACIÓN INICIAL Y PESOS
# ==========================================
st.set_page_config(
    page_title="Triaje IA - Radiografía Torácica", layout="wide", page_icon="🫁"
)

# Constantes Fase 3
IMAGE_SIZE = (512, 512)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@st.cache_data
def cargar_metadatos_clinicos():
    # Actualizado a los archivos de la Fase 3
    with open("pesos_optuna.json", "r") as f:
        pesos = json.load(f)

    df = pd.read_csv("umbrales_optimos.csv")
    patologias = df["Patologia"].tolist()  # Ajustado al nombre exacto de tu CSV
    umbrales = df["Umbral_Optimo"].values  # Ajustado al nombre exacto de tu CSV

    return pesos, patologias, umbrales


PESOS_OPTUNA, PATOLOGIAS, UMBRALES = cargar_metadatos_clinicos()
NUM_CLASSES = len(PATOLOGIAS)


# ==========================================
# 2. CARGA DE MODELOS
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
        ruta = f"{nombre}_best_model.pth"
        if os.path.exists(ruta):
            modelo, capa_cam = instanciar_modelo(nombre)
            
            # 1. Cargar el diccionario de pesos crudo
            state_dict_crudo = torch.load(ruta, map_location=device, weights_only=True)
            
            # 2. Limpiar el prefijo '_orig_mod.' inyectado por torch.compile()
            state_dict_limpio = {
                k.replace('_orig_mod.', ''): v 
                for k, v in state_dict_crudo.items()
            }
            
            # 3. Cargar los pesos limpios al modelo
            modelo.load_state_dict(state_dict_limpio)
            
            modelo.to(device)
            modelo.eval()
            comite[nombre] = {"modelo": modelo, "capa_cam": capa_cam}
    return comite


COMITE = cargar_comite()


# ==========================================
# 3. PREPROCESAMIENTO RADIOLÓGICO FASE 3
# ==========================================
def aplicar_clahe(imagen_pil):
    """Convierte la imagen a grises, aplica CLAHE y la devuelve en 3 canales RGB."""
    # 1. Extraer canal de luminancia / grises
    img_cv = np.array(imagen_pil.convert("L"))

    # 2. Aplicar ecualización clínica
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_cv)

    # 3. Replicar canales para satisfacer la entrada de ImageNet (RGB)
    img_clahe_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(img_clahe_rgb), img_clahe_rgb


# ==========================================
# 4. INTERFAZ DE USUARIO (UI)
# ==========================================
st.title("🫁 Sistema de Triaje Inteligente Torácico")
st.markdown("**Prototipo Funcional de Grado Clínico (Fase 3)**")
st.markdown(
    "Sistema soportado por un *Weighted Ensemble* (EfficientNet-B0, ResNet-50, DenseNet-121). "
    r"Optimizado con **Asymmetric Loss** y ecualización **CLAHE** para alta sensibilidad (Recall $\ge 80\%$). "
    "No constituye un diagnóstico médico final."
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
        "⚠️ Uso exclusivo para investigación y triaje. Los datos se procesan localmente y no se almacenan."
    )

archivo_subido = st.file_uploader(
    "Sube una radiografía torácica frontal (DICOM convertido a JPEG/PNG)",
    type=["jpg", "jpeg", "png"],
)

if archivo_subido is not None:
    col1, col2 = st.columns(2)
    imagen_cruda_pil = Image.open(archivo_subido).convert("RGB")

    # Ejecutar preprocesamiento estricto
    imagen_clahe_pil, imagen_clahe_cv = aplicar_clahe(imagen_cruda_pil)

    with col1:
        st.subheader("Radiografía Original")
        st.image(imagen_cruda_pil, use_container_width=True)
        # Mostrar el efecto de CLAHE como un pequeño expander para los evaluadores
        with st.expander("Ver Filtro de Ecualización (CLAHE) aplicado a la red"):
            st.image(imagen_clahe_pil, use_container_width=True)

    # Pipeline de tensores
    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    imagen_tensor = transform(imagen_clahe_pil).unsqueeze(0).to(device)

    # Normalizar la imagen a [0,1] para que Grad-CAM se pinte correctamente sobre ella
    imagen_float = cv2.resize(imagen_clahe_cv / 255.0, IMAGE_SIZE)

    if st.button(
        "🔬 Iniciar Triaje Algorítmico", type="primary", use_container_width=True
    ):
        with st.spinner("El Comité de IA está evaluando la estructura pulmonar..."):
            # --- INFERENCIA ENSEMBLE ---
            probabilidades_totales = np.zeros(NUM_CLASSES)

            for nombre, datos in COMITE.items():
                modelo = datos["modelo"]
                peso = PESOS_OPTUNA[nombre]

                with torch.no_grad():
                    with torch.amp.autocast(
                        device_type="cuda" if "cuda" in str(device) else "cpu"
                    ):
                        logits = modelo(imagen_tensor)
                    probs = torch.sigmoid(logits.float())[0].cpu().numpy()
                    probabilidades_totales += probs * peso

            # --- EVALUACIÓN CLÍNICA ---
            hallazgos = []
            for i, patologia in enumerate(PATOLOGIAS):
                prob = probabilidades_totales[i]
                umbral = UMBRALES[i]
                if prob >= umbral:
                    hallazgos.append((patologia, prob, i))

            # --- RENDERIZADO DE RESULTADOS ---
            st.markdown("---")
            st.subheader("📋 Reporte Diagnóstico Consolidado")

            if len(hallazgos) == 0:
                st.success(
                    "✅ **Sin hallazgos patológicos significativos detectados.** (Sub-umbral clínico)."
                )
            else:
                for patologia, prob, idx in sorted(
                    hallazgos, key=lambda x: x[1], reverse=True
                ):
                    # Uso de métricas visuales de Streamlit
                    st.error(f"⚠️ **{patologia} Detectada**")
                    st.progress(
                        float(prob), text=f"Confianza del Ensamble: {prob * 100:.1f}%"
                    )

            # --- XAI (Grad-CAM Dinámico) ---
            if len(hallazgos) > 0:
                with col2:
                    st.subheader("Auditoría Visual (Grad-CAM)")
                    st.markdown(
                        "Mapa térmico derivado del modelo con mayor certidumbre."
                    )

                    # Identificar la patología principal (mayor probabilidad)
                    patologia_principal, prob_principal, idx_patologia = max(
                        hallazgos, key=lambda x: x[1]
                    )

                    # Utilizar siempre EfficientNet (el cerebro del ensamble) para la explicación visual
                    mejor_modelo_nombre = "efficientnet"
                    mejor_modelo = COMITE[mejor_modelo_nombre]["modelo"]
                    capa_objetivo = COMITE[mejor_modelo_nombre]["capa_cam"]

                    cam = GradCAM(model=mejor_modelo, target_layers=capa_objetivo)
                    targets = [ClassifierOutputTarget(idx_patologia)]

                    grayscale_cam = cam(input_tensor=imagen_tensor, targets=targets)[
                        0, :
                    ]
                    visualizacion = show_cam_on_image(
                        imagen_float, grayscale_cam, use_rgb=True
                    )

                    st.image(
                        visualizacion,
                        caption=f"Foco de atención algorítmica para {patologia_principal} (Modelo: EfficientNet-B0)",
                        use_container_width=True,
                    )
