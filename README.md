# Análisis automatizado de imágenes médicas para la detección de patologías respiratorias mediante *deep learning*

Trabajo de investigación del Curso de Especialización en Machine Learning, Universidad de El Salvador, Facultad de Ingeniería y Arquitectura, Escuela de Ingeniería de Sistemas Informáticos (2026).

Instructor: Ing. Bladimir Díaz

Equipo:
Carlos David Ascencio Abarca - AA15009
Eduardo David Rodríguez Ventura - RV15001
Bryan Aldair Villalobos Mónchez - VM15018

## Propósito y alcance

Este proyecto implementa y evalúa un clasificador experimental multietiqueta de radiografías de tórax para neumonía, derrame pleural, tuberculosis y edema pulmonar. Compara seis arquitecturas convolucionales inicializadas con pesos de ImageNet o TorchXRayVision, con particiones agrupadas por paciente, calibración de probabilidades y umbrales definidos en un conjunto separado.

La cohorte retrospectiva institucional comprende 1,859 imágenes de pacientes de 15 años o más (1,516 grupos de pacientes), divididas en desarrollo (1,190), calibración (297) y prueba interna (372). Las etiquetas proceden de informes radiológicos procesados mediante PLN y no cuentan con verificación independiente por especialistas; las etiquetas no confirmadas se enmascaran durante el análisis. Los datos clínicos no se distribuyen con este repositorio.

## Resultados principales

ImageNet fue la familia seleccionada según PR-AUC macro del *ensemble* en calibración (0.5307; TorchXRayVision: 0.4929). En la prueba interna, el *ensemble* ImageNet alcanzó PR-AUC macro de 0.5853 y AUROC macro de 0.8274. La PR-AUC por etiqueta fue 0.7189 para neumonía, 0.5832 para derrame pleural, 0.5441 para tuberculosis y 0.4950 para edema. La comparación pareada por paciente no demostró superioridad concluyente entre familias.

En el contraste externo sin recalibración con 5,412 radiografías públicas NIH, la PR-AUC fue 0.0170 para neumonía, 0.2869 para derrame pleural y 0.0781 para edema; NIH no aporta etiquetas de tuberculosis en este protocolo. Esta caída evidencia una limitación importante de generalización entre dominios. Las salidas numéricas y sus intervalos de confianza se encuentran en `runs/experimento_01/stats/`; las salidas incrustadas en el notebook son históricas.

## Instalación

Se requiere Python 3.10 o superior. Desde la raíz del proyecto:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Para GPU, instale la variante de PyTorch compatible con su versión de CUDA desde el selector oficial de PyTorch.

## Ejecución

### Notebook de investigación y entrenamiento

Prepare, con autorización institucional, `export.csv` y las imágenes PNG en `data/raw/`. Abra el notebook y ajuste `PROJECT_ROOT` a la carpeta que contiene esos datos:

```bash
jupyter lab notebook.ipynb
```

Ejecute las celdas en orden para auditar los datos, entrenar y evaluar los modelos. El experimento escribe modelos y estadísticas en `runs/<RUN_ID>/`; cambie `RUN_ID` en el notebook al iniciar una cohorte o protocolo nuevo, para preservar los artefactos existentes.

### Aplicación demostrativa

Con los modelos y manifiestos disponibles en `runs/experimento_01/models/`:

```bash
streamlit run app.py
```

La interfaz compara ambas familias y permite explorar puntuaciones y mapas Grad-CAM con imágenes DICOM, PNG o JPG. Si los artefactos están en otra ubicación, configure `XRAY_IMAGENET_MODEL_DIR` o `XRAY_XRV_MODEL_DIR` antes de iniciar Streamlit.

## Estructura relevante

- `notebook.ipynb`: flujo de análisis, entrenamiento, calibración y evaluación.
- `src/proyecto/`: carga y auditoría de datos, preprocesamiento, modelos, entrenamiento, calibración y métricas.
- `runs/experimento_01/stats/`: tablas reproducibles de la ejecución evaluada.
- `app.py`: interfaz interactiva de demostración.

> **Uso exclusivamente investigativo y educativo.** El prototipo no está validado clínicamente y no debe utilizarse para diagnóstico, cribado, triaje ni decisiones asistenciales. Las puntuaciones y los mapas Grad-CAM no constituyen evidencia clínica ni localización diagnóstica.
