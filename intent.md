# INTENT — YOLOv5-OBB (Oriented Bounding Box Detection)

> **Artefacto 1 de 3 del runbook AI-Native SDLC** · `intent.md` → [`specs.md`](./specs.md) → [`plan.md`](./plan.md)
>
> Este documento responde **por qué** existe el proyecto y **qué resultado** se persigue. No describe implementación.
> Es la fuente de verdad que humanos y agentes de IA deben consultar antes de proponer cualquier cambio.

| Campo | Valor |
|---|---|
| Repositorio | `obaruch/yolov5_obb` (fork de `hukaixuan19970627/yolov5_obb`) |
| Base tecnológica | Ultralytics YOLOv5 v6.0 + Circular Smooth Label (CSL) |
| Licencia | GPL-3.0 (heredada de YOLOv5) |
| Estado | Brownfield — código existente, funcional, sin suite de tests ni CI |
| Última revisión | 2026-09-29 |
| Owner | Omar Baruch Morón López |

---

## 1. Declaración de intención (una frase)

Disponer de un pipeline **reproducible** de entrenamiento, validación e inferencia de **detección de objetos orientados (OBB)** sobre imágenes aéreas / de alta resolución, que pueda ejecutarse tanto en una estación local como en el **clúster GPU IBM LSF (`bsub`)**, y adaptarse a **datasets propios** (p. ej. exportados desde Roboflow) sin modificar el núcleo del modelo.

## 2. Problema

Los detectores estándar (YOLOv5 original) producen cajas horizontales (HBB). En imágenes aéreas, satelitales o de inspección los objetos (vehículos, barcos, aviones, piezas) aparecen **rotados y densamente empaquetados**; las HBB se solapan, incluyen mucho fondo y degradan la precisión y el NMS.

Además, en el contexto de este fork:

- El código original tiene **rutas absolutas y defaults personales** (`/media/test/...`, `--device 1`, `--batch-size 128`) que impiden ejecutarlo "tal cual".
- El entrenamiento real se hace en un **clúster con scheduler LSF**, con módulos (`conda`, `cuda/10.2.89`, `gcc/7.5.0`) y proxy, lo que no está documentado en el upstream.
- Las extensiones C++/CUDA (`utils/nms_rotated`, `DOTA_devkit/polyiou`) deben compilarse contra la versión de CUDA/PyTorch del entorno; es el punto de fallo más frecuente.
- No hay tests, CI ni criterios de aceptación explícitos: cualquier cambio (humano o de IA) puede romper silenciosamente la codificación de ángulos o el formato de etiquetas.

## 3. Resultados esperados (Outcomes)

| ID | Outcome | Cómo se reconoce |
|---|---|---|
| O1 | Entrenar un modelo OBB sobre un dataset propio desde un único comando / job `bsub`. | `run.bsub` termina con `runs/train/<name>/weights/best.pt` y `results.csv`. |
| O2 | Evaluar el modelo con métricas HBB (durante entrenamiento) y OBB (DOTA devkit). | `val.py --save-json` + `dota_evaluation_task1.py` producen mAP@0.5 por clase. |
| O3 | Inferir sobre imágenes/vídeo y obtener polígonos de 4 puntos + clase + confianza. | `detect.py --save-txt --save-conf` genera `labels/*.txt` y visualizaciones. |
| O4 | Reproducibilidad del entorno (local, Docker, clúster). | Instalación documentada; extensión `nms_rotated_ext` compila e importa. |
| O5 | Base segura para evolución asistida por IA. | Specs trazables + plan con gates de verificación; los agentes no rompen contratos. |

## 4. Métricas de éxito

| Métrica | Línea base (upstream, README) | Objetivo del fork |
|---|---|---|
| OBB mAP@0.5 DOTAv1.0 (yolov5m, 1024px) | 77.3 | No regresar > 1 pt al reproducir con pesos publicados. |
| OBB mAP@0.5 DOTAv1.5 (yolov5m, 1024px) | 73.2 | Ídem. |
| Demo `dataset/dataset_demo` (val OBB) | mAP 0.667 (GetStart.md) | Reproducible en ≤ 5 min en 1 GPU. |
| Dataset propio (Roboflow) | — (a definir tras primer baseline) | Establecer baseline y mejorar iterativamente. |
| Tiempo "clone → primer entrenamiento" en clúster | No documentado | ≤ 30 min siguiendo el runbook. |
| Jobs `bsub` fallidos por entorno | No medido | 0 tras aplicar el plan (fase de hardening). |

## 5. Stakeholders

| Rol | Interés |
|---|---|
| Owner / ML Engineer | Entrenar y evaluar modelos OBB sobre datos propios en el clúster. |
| Equipo AI Initiative (cola `rb_ai_initiative`) | Uso eficiente de GPU compartida; jobs predecibles. |
| Consumidores del modelo | Reciben polígonos orientados (formato DOTA) para aplicaciones downstream. |
| Agentes de IA (Claude Code u otros) | Implementan cambios acotados siguiendo `specs.md` y `plan.md`. |
| Upstream (hukaixuan / Ultralytics) | Compatibilidad y atribución (GPL-3.0). |

## 6. Alcance

### Dentro del alcance
- Entrenamiento, validación, inferencia y exportación del modelo YOLOv5-OBB (n/s/m/l/x).
- Formato de etiquetas DOTA (`x1 y1 ... x4 y4 classname difficult`) y conversión a rbox + CSL.
- Utilidades DOTA_devkit: split de imágenes grandes, merge de resultados, evaluación OBB.
- Ejecución en clúster LSF (`run.bsub`), DDP multi-GPU (`sh/ddp_train.sh`) y Docker.
- Documentación operativa y criterios de verificación.

### Fuera del alcance (Non-goals)
- Reescribir la arquitectura o migrar a Ultralytics v8+/YOLO-OBB nativo (se evaluaría en un intent separado).
- Detección HBB genérica (COCO) — se mantiene sólo por compatibilidad heredada.
- Servicio de inferencia productivo (la API Flask en `utils/flask_rest_api` es ejemplo, no producto).
- Etiquetado de datos o gestión del dataset en Roboflow.
- Soporte oficial para Windows (upstream lo desaconseja).

## 7. Restricciones

- **Técnicas:** Python ≥ 3.7, PyTorch ≥ 1.7, CUDA ≥ 9.0; extensiones CUDA compiladas localmente; en el clúster: `cuda/10.2.89`, `gcc/7.5.0`, `conda/4.9.2`.
- **Modelo:** la cabeza `Detect` emite `nc + 5 + 180` canales por anchor (180 clases de ángulo CSL). Cambiar esto invalida todos los checkpoints.
- **Recursos:** jobs en cola `rb_ai_initiative`, 1 GPU, 2 GB RAM/core, wallclock 2 h en la configuración actual.
- **Legal:** GPL-3.0 — todo derivado distribuido debe mantener la licencia y la atribución.
- **Datos:** los datasets propios no se versionan en el repo; sólo `dataset/dataset_demo` (1 imagen DOTA).

## 8. Principios y guardrails para agentes de IA

1. **Intent → Specs → Plan → Código.** Ningún cambio de código sin requisito (`FR-*`/`NFR-*`) y tarea (`T-*`) asociados.
2. **No romper contratos:** formato de etiquetas, rango de ángulo `θ ∈ [-π/2, π/2)`, layout de salida `(n, [xylsθ, conf, cls])`, estructura de checkpoints.
3. **Cambios mínimos y reversibles;** un PR por tarea del plan.
4. **Verificar antes de afirmar:** toda tarea cierra con evidencia (comando + salida) según los gates de `plan.md`.
5. **Human-in-the-loop** obligatorio para: cambios en `models/yolo.py`, `utils/loss.py`, `utils/rboxs_utils.py`, `utils/general.py::non_max_suppression_obb`, hiperparámetros por defecto y cualquier cosa que consuma horas de GPU.
6. **No introducir secretos, rutas personales ni datos** en el repo; no subir pesos grandes a git.
7. **Atribución:** respetar GPL-3.0 y créditos a Ultralytics, CSL, OBBDetection y DOTA_devkit.

## 9. Supuestos y preguntas abiertas

| # | Supuesto / pregunta | Acción |
|---|---|---|
| A1 | El dataset objetivo es el export Roboflow referido en `run.bsub` (`../datasets/roboflow/data.yaml`) con imágenes de ~3840 px. | Confirmar clases, nº de imágenes y formato de etiqueta (DOTA vs YOLO-OBB). |
| A2 | `weights/yolov5n.pt` en el repo pesa exactamente 1 MiB; el oficial pesa ~3.8 MB → probablemente **truncado**. | Verificar carga con `torch.load`; si falla, descargar con `data/scripts/download_weights.sh`. |
| A3 | `--img 3840` con `batch 1` en una GPU puede exceder memoria o el wallclock de 2 h. | Decidir entre split (DOTA_devkit) vs entrenamiento a resolución completa. |
| A4 | El clúster usa CUDA 10.2 → PyTorch compatible ≤ 1.12. | Fijar versiones en un `environment.yml` (ver plan). |
| Q1 | ¿Se requiere exportación (ONNX/TensorRT) para despliegue? | Pendiente del owner. |
| Q2 | ¿Métrica objetivo principal: mAP@0.5 OBB o mAP@0.5:0.95? | Pendiente del owner. |

## 10. Criterio de "intent cumplido"

El intent se considera satisfecho cuando O1–O4 están verificados con evidencia en el clúster y localmente, las specs `FR-*` críticas tienen pruebas automáticas (O5), y existe un baseline documentado de mAP OBB sobre el dataset propio.
