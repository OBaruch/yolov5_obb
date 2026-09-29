# SPECS — YOLOv5-OBB

> **Artefacto 2 de 3 del runbook AI-Native SDLC** · [`intent.md`](./intent.md) → `specs.md` → [`plan.md`](./plan.md)
>
> Este documento define **qué** debe hacer el sistema, con contratos verificables. Cada requisito tiene un ID estable
> (`FR-*`, `NFR-*`, `IF-*`, `DC-*`) que `plan.md`, commits y PRs deben referenciar.
> Describe el comportamiento **actual** del código (brownfield) y marca con 🔧 lo que el plan debe corregir o añadir.

---

## 1. Contexto del sistema

```
                 ┌──────────────────────── DOTA_devkit ─────────────────────────┐
 imágenes +      │ ImgSplit_multi_process.py  (split 1024/gap200)               │
 labelTxt DOTA ─►│                                                              │
                 └───────────────┬──────────────────────────────────────────────┘
                                 ▼
 data/*.yaml ──► train.py ──► utils/datasets.py ─(poly2rbox + CSL)─► models/yolo.py ──► utils/loss.py
 data/hyps/obb/*.yaml            │                                     (Detect: nc+5+180)   (box/obj/cls/θ)
                                 ▼
                   runs/train/<name>/weights/{best,last}.pt
                                 │
            ┌────────────────────┼─────────────────────┬──────────────────┐
            ▼                    ▼                     ▼                  ▼
        val.py (HBB mAP     detect.py (polígonos   export.py          hubconf.py
        + JSON OBB)         + imágenes/txt)        (TS/ONNX/...)      (torch.hub)
            │
            ▼
 tools/TestJson2VocClassTxt.py ─► DOTA_devkit/ResultMerge_multi_process.py ─► dota_evaluation_task1.py (OBB mAP)
```

### Componentes

| Componente | Ruta | Responsabilidad |
|---|---|---|
| Entrenamiento | `train.py` | Bucle de entrenamiento, DDP, EMA, early stopping, evolución de hiperparámetros, logging. |
| Validación | `val.py` | mAP HBB (P/R/mAP@.5/.5:.95), exporta JSON con polígonos OBB, modo `speed`/`study`. |
| Inferencia | `detect.py` | Imágenes/vídeo/stream/webcam → polígonos rotados, dibujo, `.txt`. |
| Exportación | `export.py` | TorchScript, ONNX, CoreML, OpenVINO, TF SavedModel/PB/TFLite/TFJS, TensorRT. |
| Modelo | `models/yolo.py`, `models/common.py`, `models/yolov5{n,s,m,l,x}.yaml` | Backbone CSP + PANet + cabeza `Detect` OBB. |
| Utilidades OBB | `utils/rboxs_utils.py` | `poly2rbox`, `rbox2poly`, `poly2hbb`, `poly_filter`, CSL gaussiana, `regular_theta`. |
| Pérdida | `utils/loss.py::ComputeLoss` | IoU (HBB) + objectness + clase + ángulo (BCE sobre CSL). |
| NMS rotado | `utils/general.py::non_max_suppression_obb`, `utils/nms_rotated/` | NMS con IoU rotado (extensión C++/CUDA). |
| Datos | `utils/datasets.py`, `utils/augmentations.py` | Carga `images/`↔`labelTxt/`, mosaic, mixup, rotación, flips. |
| DOTA devkit | `DOTA_devkit/` | Split, merge, evaluación OBB (task1) y HBB (task2), `polyiou` (SWIG). |
| Herramientas | `tools/TestJson2VocClassTxt.py`, `tools/Xml2Txt.py` | JSON → txt por clase (Task1_*.txt); XML → labelTxt. |
| Operación | `run.bsub`, `sh/ddp_train.sh`, `Dockerfile` | Ejecución en LSF, multi-GPU, contenedor. |

---

## 2. Contratos de datos (DC)

### DC-1 · Etiqueta de entrada (formato DOTA poly)
- Un `.txt` por imagen en `<split>/labelTxt/`, mismo *stem* que la imagen en `<split>/images/` (mapeo en `utils/datasets.py::img2label_paths`).
- Una línea por objeto: `x1 y1 x2 y2 x3 y3 x4 y4 classname difficult` (píxeles absolutos, float).
- `classname` ∈ `names` del YAML del dataset; `difficult` ∈ {0,1}.
- Las líneas de cabecera DOTA (`imagesource:`, `gsd:`) deben ser ignoradas.

### DC-2 · Representación interna rbox
- `[cx, cy, l, s, θ]` en píxeles: `l` = lado largo, `s` = lado corto (formato *long-edge*).
- `θ ∈ [-π/2, π/2)` (radianes) internamente; `angle ∈ [0, 180)` para la clase CSL.
- Conversión: `cv2.minAreaRect` → long-edge → `regular_theta` (`utils/rboxs_utils.py::poly2rbox`).

### DC-3 · Etiqueta CSL (Circular Smooth Label)
- Vector de longitud `cls_theta` (=180) generado con ventana gaussiana circular de radio `csl_radius` (hyp).
- Tensor de targets por batch: `[img_index, cls_id, cx, cy, l, s, θ, csl_0..csl_179]` → `7 + cls_theta` columnas.

### DC-4 · Salida de la cabeza del modelo
- Por anchor: `no = nc + 5 + 180` → `[cx, cy, w, h, obj, cls_0..cls_{nc-1}, θ_0..θ_179]`.
- 3 niveles (P3/P4/P5), 3 anchors por nivel.

### DC-5 · Salida post-NMS
- `non_max_suppression_obb` → lista por imagen de tensores `(n, 7)`: `[cx, cy, l, s, θ, conf, cls]`, `θ ∈ [-π/2, π/2)`.
- `max_det` por defecto 1500 (NMS) / 1000 (`detect.py`); `max_wh = 4096`.

### DC-6 · Salida de detección (`detect.py --save-txt`)
- `runs/detect/<name>/labels/<stem>.txt`, una línea por objeto: `cls x1 y1 x2 y2 x3 y3 x4 y4 [conf]` en píxeles de la imagen original.

### DC-7 · JSON de predicciones (`val.py --save-json`)
- `runs/val/<name>/<weights>_obb_predictions.json` con polígono, bbox HBB, `category_id`, `score`, `image_id`.
- Consumido por `tools/TestJson2VocClassTxt.py` → `Task1_<classname>.txt` (formato de envío DOTA).

### DC-8 · YAML de dataset (`data/*.yaml`)
- Claves: `path`, `train`, `val`, `test` (opcional), `nc`, `names`. Rutas relativas a `path`.
- `len(names) == nc`.

### DC-9 · YAML de hiperparámetros (`data/hyps/obb/*.yaml`)
- Claves obligatorias OBB además de las YOLOv5: `theta`, `theta_pw`, `cls_theta`, `csl_radius`.
- `degrees` admite hasta 180 (rotación de augmentación).

### DC-10 · Checkpoint
- `.pt` con `model`, `ema`, `optimizer`, `epoch`, `best_fitness`, `wandb_id`, `date` (estándar YOLOv5 v6.0); `strip_optimizer` al final.

---

## 3. Requisitos funcionales (FR)

| ID | Requisito | Criterio de aceptación (Given / When / Then) | Estado |
|---|---|---|---|
| FR-1 | Entrenar un modelo OBB desde pesos preentrenados o desde cero. | **Given** un YAML válido (DC-8) y hyp OBB (DC-9) **When** `python train.py --data D --weights W --hyp H --epochs E --img S --device 0` **Then** se crean `runs/train/<name>/weights/{last,best}.pt`, `results.csv` y `results.png`, con las 4 pérdidas (`box, obj, cls, theta`) registradas. | ✅ |
| FR-2 | Reanudar entrenamiento. | **When** `--resume [ckpt]` **Then** continúa desde la época guardada con el mismo optimizador/EMA. | ✅ |
| FR-3 | Entrenamiento multi-GPU DDP. | **When** `python -m torch.distributed.launch --nproc_per_node N train.py --device 0,..,N-1` **Then** entrena con `--sync-bn` opcional y sólo RANK 0 guarda. | ✅ |
| FR-4 | Validación HBB durante y tras entrenamiento. | **When** `val.py --task val` **Then** imprime P, R, mAP@.5, mAP@.5:.95 global y por clase (`--verbose`). | ✅ |
| FR-5 | Exportar predicciones OBB a JSON y a formato DOTA Task1. | **When** `val.py --save-json` + `tools/TestJson2VocClassTxt.py` **Then** se genera DC-7 y un `Task1_<cls>.txt` por clase. | ✅ |
| FR-6 | Evaluación OBB oficial. | **When** `dota_evaluation_task1.py --detpath --annopath --imagesetfile` **Then** imprime `map` y `classaps`. Demo: `map ≈ 0.667`. | ✅ |
| FR-7 | Split de imágenes de alta resolución y merge de resultados. | **When** `ImgSplit_multi_process.py` (subsize 1024, gap 200, rate 1.0) **Then** genera `images/`+`labelTxt/` por parche; `ResultMerge_multi_process.py` fusiona con NMS poligonal a coordenadas originales. | ✅ |
| FR-8 | Inferencia sobre imagen/directorio/vídeo/stream/webcam. | **When** `detect.py --weights W --source S` **Then** guarda imágenes con polígonos rotados y, con `--save-txt`, DC-6. Soporta `--classes`, `--agnostic-nms`, `--half`, `--hide-labels`, `--hide-conf`. | ✅ |
| FR-9 | NMS rotado en GPU y CPU. | **Given** extensión `nms_rotated_ext` compilada **When** se llama `non_max_suppression_obb` **Then** devuelve DC-5 en CPU o CUDA. | ✅ |
| FR-10 | Exportar el modelo a formatos de despliegue. | **When** `export.py --weights W --include onnx` **Then** genera el artefacto; la salida conserva `nc+5+180` canales. Soporte OBB en post-proceso de formatos no-PyTorch: **no verificado**. | ⚠️ 🔧 |
| FR-11 | Ejecución en clúster IBM LSF. | **When** `bsub < run.bsub` **Then** el job carga módulos (`conda`, `cuda/10.2.89`, `gcc/7.5.0`), activa `yolov5`, activa proxy y ejecuta `train.py`. | ✅ (parametrización 🔧) |
| FR-12 | Evolución de hiperparámetros. | **When** `--evolve G` **Then** genera `evolve.csv` y `hyp_evolve.yaml`. | ✅ (no probado en OBB) |
| FR-13 | Entrenamiento single-class. | **When** `--single-cls` **Then** entrena con `nc=1` (upstream advierte peor resultado que `nc=2`). | ✅ |
| FR-14 | Carga vía `torch.hub`. | **When** `torch.hub.load(repo, 'custom', path=W)` **Then** devuelve modelo listo para inferencia. | ⚠️ no verificado para OBB |
| FR-15 | Conversión de anotaciones XML (VOC/HRSC) a labelTxt. | **When** `tools/Xml2Txt.py` **Then** genera DC-1. | ✅ |
| FR-16 | 🔧 Configuración sin rutas personales. | **Given** un clon limpio **When** se ejecutan `train.py`/`val.py`/`detect.py` sin argumentos de ruta **Then** usan defaults relativos al repo (`data/yolov5obb_demo.yaml`, `dataset/dataset_demo`, `--device 0`) y no `/media/test/...`. | 🔧 |
| FR-17 | 🔧 Job LSF parametrizable. | **When** `bsub -env "DATA=...,IMG=...,EPOCHS=..." < run.bsub` (o equivalente) **Then** no hace falta editar el script para cambiar dataset/hiperparámetros; logs a `logs/%J.{out,err}`. | 🔧 |
| FR-18 | 🔧 Pesos preentrenados íntegros. | **When** se usa `--weights weights/yolov5n.pt` **Then** `torch.load` no falla; si el fichero falta o está truncado, se descarga automáticamente (`utils/downloads.py::attempt_download`). | 🔧 |

## 4. Requisitos no funcionales (NFR)

| ID | Categoría | Requisito | Verificación |
|---|---|---|---|
| NFR-1 | Precisión | Con pesos publicados yolov5m DOTAv1.5 @1024, OBB mAP@0.5 test ≥ 72.2 (baseline 73.2 − 1). | Pipeline FR-5→FR-6 + servidor DOTA. |
| NFR-2 | Rendimiento | Inferencia yolov5m b1 @1024 ≤ 20 ms en GPU clase 2080Ti (incl. NMS). | `val.py --task speed --batch 1`. |
| NFR-3 | Reproducibilidad | Mismo commit + mismo entorno + `init_seeds(0)` → métricas dentro de ±0.5 mAP. | Dos ejecuciones del demo. |
| NFR-4 | Portabilidad | Linux (Ubuntu 16.04/18.04+), Python 3.7–3.9, PyTorch 1.7–1.12, CUDA 10.0–11.3; Docker `nvcr.io/nvidia/pytorch:21.10-py3`. | Matriz de instalación (plan F1). |
| NFR-5 | Compilación | `utils/nms_rotated` y `DOTA_devkit/polyiou` compilan con `python setup.py develop` / `swig` + `build_ext --inplace`. | Smoke test de import. |
| NFR-6 | Recursos clúster | Un job estándar cabe en 1 GPU, ≤ wallclock configurado; OOM se detecta y reporta. | Log LSF. |
| NFR-7 | Calidad de código | flake8 según `setup.cfg` (max-line 120); sin nuevas violaciones en archivos tocados. | `flake8 <files>`. |
| NFR-8 | Testabilidad | 🔧 Tests unitarios para `utils/rboxs_utils.py`, `non_max_suppression_obb` y carga de dataset; ejecutables en CPU en < 2 min. | `pytest` (plan F2). |
| NFR-9 | Observabilidad | TensorBoard siempre; W&B opcional; `results.csv` con pérdidas y métricas por época. | Inspección de `runs/`. |
| NFR-10 | Seguridad / higiene | Sin credenciales, rutas personales ni datasets en git; pesos > 5 MB fuera de git (release o descarga). | Revisión de PR + secret scanning. |
| NFR-11 | Licencia | Cabeceras y `LICENSE` GPL-3.0 intactos; atribución en README. | Revisión de PR. |

## 5. Interfaces (IF)

| ID | Interfaz | Argumentos clave (default actual) | Notas |
|---|---|---|---|
| IF-1 | `train.py` CLI | `--weights` (`weights/yolov5n.pt`), `--data` (`data/DroneVehicle_poly.yaml` 🔧), `--hyp` (`data/hyps/obb/hyp.finetune_dota.yaml`), `--epochs` (5), `--batch-size` (128 🔧), `--img` (840), `--device` (`1` 🔧), `--cache`, `--rect`, `--resume`, `--evolve`, `--sync-bn`, `--freeze`, `--patience` (100), `--project`/`--name`/`--exist-ok` | 🔧 defaults a normalizar (FR-16). |
| IF-2 | `val.py` CLI | `--data`, `--weights`, `--batch-size` (8), `--img` (1024), `--conf-thres` (0.01), `--iou-thres` (0.4), `--task` (`val`/`train`/`test`/`speed`/`study`), `--save-json`, `--half`, `--verbose` | `--device` default `1` 🔧. |
| IF-3 | `detect.py` CLI | `--weights`, `--source` (ruta absoluta 🔧), `--img` (840), `--conf-thres` (0.25), `--iou-thres` (0.2), `--max-det` (1000), `--save-txt`, `--save-conf`, `--save-crop`, `--classes`, `--agnostic-nms`, `--half` | `--device` default `3` 🔧. |
| IF-4 | `export.py` CLI | `--weights`, `--include` (`torchscript onnx`), `--opset` (12), `--dynamic`, `--simplify`, `--half` | Default `--data data/dotav15_poly.yaml`. |
| IF-5 | `run.bsub` | `-J`, `-q rb_ai_initiative`, `-W 2:00`, `-n 1`, `-M 2048`, `-gpu "num=1"` | Comando fijo: `--data ../datasets/roboflow/data.yaml --epochs 10 --batch-size 1 --img 3840`. |
| IF-6 | API Python | `poly2rbox(polys, num_cls_thata, radius, use_pi, use_gaussian)`, `rbox2poly(obboxes)`, `poly2hbb`, `non_max_suppression_obb(prediction, conf_thres, iou_thres, ...)` | Contratos DC-2..DC-5. |
| IF-7 | DOTA devkit CLI | `ResultMerge_multi_process.py --scrpath --dstpath`; `dota_evaluation_task1.py --detpath --annopath --imagesetfile`; `TestJson2VocClassTxt.py --json_path --save_path` | |

## 6. Configuraciones soportadas

| Dataset YAML | nc | Uso |
|---|---|---|
| `data/yolov5obb_demo.yaml` | 16 | Demo con `dataset/dataset_demo` (1 imagen, P0032) sin split. |
| `data/yolov5obb_demo_split.yaml` | 16 | Demo con dataset partido. |
| `data/dotav1_poly.yaml` / `data/dotav15_poly.yaml` | 15 / 16 | DOTA oficial (rutas absolutas del autor 🔧). |
| `data/DroneVehicle_poly.yaml` | 2 | DroneVehicle (rutas absolutas 🔧). |
| `../datasets/roboflow/data.yaml` (externo) | ? | Dataset propio usado en `run.bsub` (ver A1 en intent). |

| Hyp | Uso |
|---|---|
| `hyp.finetune_dota.yaml` | Default: fine-tune DOTA, `degrees 180`, `mosaic 0.75`, `mixup 0.1`, `csl_radius 2.0`. |
| `hyp.finetune_dota_CloseAug.yaml` | Sin augmentaciones fuertes (cierre de entrenamiento). |
| `hyp.finetune_DroneVehicle.yaml` | DroneVehicle. |
| `hyp.paper.yaml` | Reproducción de paper. |

## 7. Defectos y deuda conocidos (entradas para el plan)

| ID | Ubicación | Descripción | Severidad |
|---|---|---|---|
| KD-1 | `train.py`, `val.py`, `detect.py` (argparse) | Defaults con rutas absolutas personales y GPUs `1`/`3`; `--batch-size 128`. | Alta (rompe "clone & run"). |
| KD-2 | `weights/yolov5n.pt` | Fichero de exactamente 1 048 576 bytes; probable truncado (oficial ≈ 3.8 MB). | Alta. |
| KD-3 | `utils/datasets.py` (~L644) | `labels_mask`: compara `rboxes[:, 0] < img.shape[0]` (debería ser `[:, 1]` para `cy`) y la precedencia `& ... \| ...` hace que `(s > 5)` anule los demás filtros. | Media — requiere validación humana antes de corregir (afecta métricas). |
| KD-4 | `models/yolo.py::Detect`, `utils/general.py::non_max_suppression_obb` | 180 clases de ángulo *hardcodeadas*, mientras que `cls_theta` es configurable en hyp → incoherencia si se cambia. | Media. |
| KD-5 | Repo | Sin tests, sin CI, sin `.gitignore` (riesgo de subir `runs/`, pesos, datasets). | Alta para AI-native. |
| KD-6 | `run.bsub` | Parámetros fijos; `echo $('start')` inválido; sin `-o/-e`; `--img 3840` con 2 h/2 GB puede fallar. | Media. |
| KD-7 | `requirements.txt` | Versiones sin techo; incompatibilidades posibles con CUDA 10.2 del clúster. | Media. |
| KD-8 | `export.py`, `hubconf.py` | Post-proceso OBB no validado fuera de PyTorch. | Baja/Media. |
| KD-9 | `test.txt`, `sh/ddp_train.sh` | Notas de resultados y script con rutas del autor original, sin contexto. | Baja. |

## 8. Matriz de trazabilidad Intent → Specs

| Outcome | Requisitos |
|---|---|
| O1 Entrenar desde un comando / bsub | FR-1, FR-2, FR-3, FR-11, FR-16, FR-17, FR-18, NFR-6 |
| O2 Evaluar HBB + OBB | FR-4, FR-5, FR-6, FR-7, NFR-1 |
| O3 Inferencia | FR-8, FR-9, FR-10, FR-14, NFR-2 |
| O4 Entorno reproducible | NFR-3, NFR-4, NFR-5, KD-7 |
| O5 Evolución segura con IA | NFR-7, NFR-8, NFR-10, NFR-11, KD-5, §8 de intent |

## 9. Definition of Done (por requisito)

Un requisito pasa a ✅ cuando:
1. Existe evidencia reproducible (comando + salida) en el PR que lo toca.
2. Si es 🔧, hay test automático o smoke test en el gate correspondiente de `plan.md`.
3. No se violan DC-1..DC-10 (tests de contrato verdes).
4. flake8 sin nuevas violaciones en archivos modificados.
5. Revisión humana aprobada si toca archivos marcados como sensibles en `intent.md §8`.
