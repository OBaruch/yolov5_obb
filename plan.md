# PLAN — YOLOv5-OBB

> **Artefacto 3 de 3 del runbook AI-Native SDLC** · [`intent.md`](./intent.md) → [`specs.md`](./specs.md) → `plan.md`
>
> Este documento define **cómo y en qué orden** se ejecuta el trabajo: fases, tareas atómicas (`T-*`) trazadas a
> requisitos (`FR/NFR/KD`), gates de verificación, responsabilidades humano/agente y riesgos.
> Cada tarea = 1 rama/PR pequeño. Ninguna tarea se ejecuta sin su requisito en `specs.md`.

---

## 1. Ciclo AI-Native SDLC (runbook estándar)

```
 ┌────────┐   ┌────────┐   ┌────────┐   ┌───────────┐   ┌──────────┐   ┌─────────┐   ┌──────────┐
 │ INTENT │──►│ SPECS  │──►│  PLAN  │──►│ IMPLEMENT │──►│  VERIFY  │──►│ REVIEW  │──►│ OPERATE  │
 │ humano │   │ IA+hum │   │ IA+hum │   │  agente   │   │ agente+CI│   │ humano  │   │ humano   │
 └────────┘   └────────┘   └────────┘   └───────────┘   └──────────┘   └─────────┘   └────┬─────┘
      ▲                                                                                   │
      └──────────────────── feedback: métricas, fallos de job, nuevas preguntas ──────────┘
```

| Etapa | Entrada | Salida | Responsable | Gate de salida |
|---|---|---|---|---|
| 1. Intent | Necesidad de negocio | `intent.md` | Owner | Outcomes y non-goals aprobados. |
| 2. Specs | `intent.md` + código | `specs.md` | Agente redacta, owner aprueba | Cada outcome trazado a ≥1 FR/NFR con criterio de aceptación. |
| 3. Plan | `specs.md` | `plan.md` | Agente redacta, owner aprueba | Cada tarea referencia un requisito y tiene verificación. |
| 4. Implement | Tarea `T-*` | Rama + commits | Agente | Diff mínimo; sólo archivos de la tarea. |
| 5. Verify | Rama | Evidencia (logs, tests) | Agente + CI | Gate de la fase en verde (§4). |
| 6. Review | PR | Aprobación | Owner (obligatorio en archivos sensibles) | Checklist §6 completo. |
| 7. Operate | `main` | Jobs en clúster, métricas | Owner | Métricas registradas; feedback a intent/specs. |

### Convenciones
- **Ramas:** `feat/T-XX-descripcion`, `fix/T-XX-...`, `docs/T-XX-...`.
- **Commits:** Conventional Commits con referencia: `fix(datasets): corrige labels_mask (T-21, KD-3)`.
- **PR:** título con `T-XX`; cuerpo con requisito, evidencia de verificación y riesgos.
- **Actualización de artefactos:** si una tarea descubre algo que contradice specs, se actualiza `specs.md` **antes** de continuar.

---

## 2. Fases y tareas

Leyenda — **Tipo:** 🤖 agente autónomo · 🤝 agente + aprobación humana · 👤 sólo humano · **GPU:** requiere horas de GPU.

### Fase 0 — Baseline y línea de salida (sin cambios de código)

| ID | Tarea | Req. | Tipo | Verificación |
|---|---|---|---|---|
| T-01 | Crear `intent.md`, `specs.md`, `plan.md`. | O5 | 🤝 | Aprobación del owner. ✅ |
| T-02 | Verificar integridad de `weights/yolov5n.pt` (`python -c "import torch; torch.load('weights/yolov5n.pt')"`). | FR-18, KD-2 | 🤖 | Salida del comando anotada en el PR/issue. |
| T-03 | Responder supuestos/preguntas abiertas A1–A4, Q1–Q2 del intent (dataset Roboflow, métrica objetivo, export). | Intent §9 | 👤 | `intent.md` actualizado. |
| T-04 | Ejecutar demo end-to-end en local o clúster (train 10 épocas demo → val → JSON → Task1 → eval) y registrar métricas baseline. | FR-1, FR-4..FR-6 | 🤝 GPU | `map` ≈ 0.667 en demo; logs guardados. |

**Gate G0:** baseline documentado; decisiones A1–A4 tomadas.

### Fase 1 — Entorno reproducible

| ID | Tarea | Req. | Tipo | Verificación |
|---|---|---|---|---|
| T-10 | Añadir `.gitignore` (`runs/`, `*.pt` salvo lista blanca, `datasets/`, `*.so`, `build/`, `__pycache__/`, `logs/`). | NFR-10, KD-5 | 🤖 | `git status` limpio tras un entrenamiento demo. |
| T-11 | Crear `environment.yml` / `requirements-lock.txt` fijando versiones compatibles con CUDA 10.2 (p. ej. torch 1.10–1.12 + cu102). | NFR-4, KD-7 | 🤝 | Instalación limpia en clúster + `torch.cuda.is_available()`. |
| T-12 | Script `scripts/build_ext.sh` que compila `utils/nms_rotated` y `DOTA_devkit/polyiou` y hace smoke import. | NFR-5 | 🤖 | `python -c "from utils.nms_rotated import obb_nms"` y `import polyiou` OK. |
| T-13 | Actualizar `docs/install.md` con sección "Clúster LSF" (módulos, conda, proxy, compilación). | O4, FR-11 | 🤖 | Seguir la guía en un entorno nuevo ≤ 30 min. |
| T-14 | Sustituir `weights/yolov5n.pt` por descarga automática/documentada si T-02 confirma truncado. | FR-18, KD-2 | 🤝 | Carga correcta; peso fuera de git. |

**Gate G1:** instalación desde cero reproducible en local + clúster; extensiones compiladas; `git status` limpio.

### Fase 2 — Red de seguridad (tests + CI)

| ID | Tarea | Req. | Tipo | Verificación |
|---|---|---|---|---|
| T-20 | Tests de contrato `tests/test_rboxs_utils.py`: `poly2rbox`↔`rbox2poly` ida y vuelta (IoU ≥ 0.99), rango `θ ∈ [-π/2, π/2)`, `l ≥ s`, CSL con pico en `angle` y longitud 180. | DC-2, DC-3, NFR-8 | 🤖 | `pytest tests/ -q` verde en CPU. |
| T-21 | Tests `non_max_suppression_obb`: shape `(n,7)`, supresión de duplicados rotados, filtro por `conf_thres` y `classes`. | DC-5, FR-9 | 🤖 | pytest verde (CPU). |
| T-22 | Test de dataset: carga `dataset/dataset_demo`, verifica `labels_out.shape[1] == 7 + cls_theta` y nº de objetos > 0. | DC-1, DC-3 | 🤖 | pytest verde. |
| T-23 | Smoke test de modelo: `Model('models/yolov5n.yaml', nc=16)` forward con `1×3×256×256`, salida `no = nc+185`. | DC-4 | 🤖 | pytest verde. |
| T-24 | CI GitHub Actions (CPU): flake8 sobre archivos cambiados + pytest + build de `nms_rotated` CPU. | NFR-7, NFR-8, KD-5 | 🤝 | Workflow verde en PR. |

**Gate G2:** CI verde; cobertura de contratos DC-2..DC-5. **A partir de aquí toda tarea que toque código debe mantener G2 verde.**

### Fase 3 — Hardening de configuración y operación en clúster

| ID | Tarea | Req. | Tipo | Verificación |
|---|---|---|---|---|
| T-30 | Normalizar defaults de `train.py`/`val.py`/`detect.py`: data demo relativo, `--device ''` (auto), `--batch-size 16`, `--source dataset/dataset_demo/images`. | FR-16, KD-1 | 🤝 | Ejecutar los 3 scripts sin argumentos en clon limpio. |
| T-31 | Parametrizar `run.bsub` (variables `DATA`, `IMG`, `EPOCHS`, `BATCH`, `WEIGHTS`, `NAME`), añadir `#BSUB -o logs/%J.out -e logs/%J.err`, corregir `echo`, `set -euo pipefail`. | FR-17, KD-6 | 🤝 | `bsub < run.bsub` con variables → job OK y logs en `logs/`. |
| T-32 | Variante `run_ddp.bsub` (multi-GPU) basada en `sh/ddp_train.sh` con rutas relativas. | FR-3 | 🤝 GPU | Job con `num=2` completa ≥ 1 época. |
| T-33 | Crear `data/custom_roboflow.yaml` plantilla (rutas relativas, `nc`/`names` según A1) y guía de conversión Roboflow → DOTA labelTxt si aplica. | FR-1, A1 | 🤝 | `train.py --data data/custom_roboflow.yaml --epochs 1` arranca. |
| T-34 | Estrategia para 3840 px: pipeline split (FR-7) + entrenamiento 1024 + merge; comparar vs full-res en memoria/tiempo. | NFR-6, A3 | 🤝 GPU | Tabla comparativa (mAP, tiempo, memoria) en `docs/`. |
| T-35 | Limpiar `test.txt` y `sh/ddp_train.sh` → mover a `docs/experiments/` con contexto. | KD-9 | 🤖 | Sin rutas personales en el repo (`grep -r /media/test`). |

**Gate G3:** job `bsub` estándar reproducible sin editar scripts; 0 rutas absolutas personales; G2 verde.

### Fase 4 — Correctitud del modelo (sensible, human-in-the-loop)

| ID | Tarea | Req. | Tipo | Verificación |
|---|---|---|---|---|
| T-40 | Corregir `labels_mask` en `utils/datasets.py` (índice `cy` y paréntesis). Primero test que reproduce el defecto. | KD-3 | 🤝 | Test rojo → verde; re-entrenar demo y comparar mAP vs baseline T-04 (no regresión). |
| T-41 | Unificar nº de clases de ángulo: derivar `180` de `hyp['cls_theta']` o documentar que es constante y validar en `train.py`. | KD-4 | 🤝 | Assert/validación + test; checkpoints existentes siguen cargando. |
| T-42 | Validar export ONNX con post-proceso OBB (decodificación + NMS en Python) y comparar con PyTorch. | FR-10, KD-8, Q1 | 🤝 | Diferencia de detecciones ≤ 1 % en demo. |
| T-43 | Validar `hubconf.py` para modelos OBB. | FR-14 | 🤖 | Test de carga `torch.hub.load(..., source='local')`. |

**Gate G4:** mAP demo ≥ baseline T-04; tests de contrato verdes; aprobación del owner.

### Fase 5 — Modelo sobre dataset propio y operación

| ID | Tarea | Req. | Tipo | Verificación |
|---|---|---|---|---|
| T-50 | Entrenamiento baseline sobre dataset propio (yolov5n → s/m según recursos). | O1, FR-1 | 👤 GPU | `best.pt` + `results.csv`. |
| T-51 | Evaluación OBB completa del baseline (FR-5 → FR-7 → FR-6) y registro en `docs/results_custom.md`. | O2 | 🤝 | mAP@0.5 OBB por clase documentado. |
| T-52 | Iteración de hiperparámetros (`--evolve` o manual: `csl_radius`, `degrees`, `mosaic`, img size). | FR-12 | 🤝 GPU | Mejora vs T-51 documentada. |
| T-53 | Reproducir NFR-1 con pesos publicados yolov5m DOTAv1.5 (opcional, si hay acceso a DOTA). | NFR-1 | 👤 GPU | mAP test ≥ 72.2. |

**Gate G5 (= intent cumplido):** O1–O4 con evidencia; baseline propio documentado; CI verde.

---

## 3. Dependencias y ruta crítica

```
T-01 ─► T-02 ─► T-14
  │      └────────────┐
  ├─► T-03 ─► T-33 ─► T-34 ─► T-50 ─► T-51 ─► T-52
  └─► T-04 ───────────────────────► T-40 (baseline para comparar)
T-10, T-11 ─► T-12 ─► T-24
T-20..T-23 ─► T-24 ─► (G2) ─► T-30, T-31, T-40, T-41, T-42
```

**Ruta crítica:** T-02 → T-03 → T-11/T-12 → T-20..T-24 → T-31 → T-33 → T-34 → T-50 → T-51.

## 4. Gates de verificación (comandos estándar)

| Gate | Comando(s) | Criterio |
|---|---|---|
| Lint | `flake8 <archivos modificados>` | Sin nuevas violaciones (config `setup.cfg`). |
| Unit/contract | `pytest tests/ -q` | 100 % verde, CPU, < 2 min. |
| Extensiones | `bash scripts/build_ext.sh` | Imports OK. |
| Smoke train | `python train.py --data data/yolov5obb_demo.yaml --weights weights/yolov5n.pt --epochs 1 --batch-size 1 --img 640 --device cpu` | Termina sin error; genera `last.pt`. |
| Smoke detect | `python detect.py --weights <last.pt> --source dataset/dataset_demo/images --img 640 --device cpu --save-txt` | Genera `labels/P0032.txt` con formato DC-6. |
| OBB eval (GPU) | Pipeline `val.py --save-json` → `TestJson2VocClassTxt.py` → `dota_evaluation_task1.py` | mAP ≥ baseline − tolerancia. |
| Clúster | `bsub < run.bsub` (parametrizado) | Exit 0, `best.pt` generado, logs en `logs/`. |

## 5. Roles: humano vs agente

| Actividad | Agente IA | Humano (owner) |
|---|---|---|
| Redactar/actualizar intent | Propone | **Decide y aprueba** |
| Redactar specs y plan | Redacta | Revisa y aprueba |
| Tareas 🤖 (docs, tests, CI, limpieza) | Implementa y verifica | Revisión ligera |
| Tareas 🤝 (config, bsub, archivos sensibles) | Implementa, adjunta evidencia | **Aprueba antes de merge** |
| Tareas 👤 / GPU largas | Prepara scripts y análisis | Lanza jobs y valida resultados |
| Cambios en `models/yolo.py`, `utils/loss.py`, `utils/rboxs_utils.py`, NMS OBB | Sólo con test previo y comparación de mAP | **Aprobación obligatoria** |

### Prompt estándar para el agente por tarea
```
Contexto: lee intent.md, specs.md y plan.md.
Tarea: T-XX (<título>). Requisitos: <FR/NFR/KD>.
Restricciones: diff mínimo; no tocar archivos fuera del alcance; respetar DC-1..DC-10.
Verificación: ejecuta los gates de la fase <N> y pega comandos + salida.
Entrega: rama feat/T-XX-..., commit convencional con referencia, resumen de riesgos.
```

## 6. Checklist de revisión de PR

- [ ] Referencia a `T-XX` y requisito(s) en título/cuerpo.
- [ ] Diff limitado al alcance de la tarea.
- [ ] Gates de la fase ejecutados con evidencia pegada.
- [ ] Contratos DC-1..DC-10 intactos (o `specs.md` actualizado y aprobado).
- [ ] Sin rutas personales, secretos, datasets ni pesos grandes.
- [ ] Licencia/atribución GPL-3.0 intacta.
- [ ] `specs.md` (estado ✅/🔧) y este plan actualizados si procede.

## 7. Riesgos y mitigaciones

| ID | Riesgo | Prob. | Impacto | Mitigación |
|---|---|---|---|---|
| R-1 | Extensión `nms_rotated` no compila con la combinación CUDA/GCC/PyTorch del clúster. | Alta | Alto | T-11 fija versiones; T-12 script de build; fallback CPU. |
| R-2 | `--img 3840` provoca OOM o supera wallclock 2 h. | Alta | Medio | T-34 split + train 1024; ajustar `-W`/`-M` en `run.bsub`. |
| R-3 | Corrección de KD-3 cambia distribución de targets y altera mAP. | Media | Alto | Test previo, comparación con baseline T-04, aprobación humana. |
| R-4 | Pesos truncados producen error críptico al inicio del job. | Alta | Medio | T-02/T-14 antes de cualquier job. |
| R-5 | Agente modifica contratos (ángulo/formato) sin darse cuenta. | Media | Alto | Tests de contrato G2 obligatorios; archivos sensibles con aprobación. |
| R-6 | Dependencias sin techo rompen instalación futura. | Media | Medio | Lockfile T-11; CI semanal (opcional). |
| R-7 | Uso de cola compartida sin control de coste GPU. | Media | Medio | Tareas GPU marcadas; smoke tests en CPU antes de lanzar jobs. |

## 8. Runbook operativo rápido (estado actual, sin cambios de código)

```bash
# 1) Entorno
conda create -n yolov5 python=3.9 -y && conda activate yolov5
pip install torch torchvision  # versión compatible con la CUDA del entorno
pip install -r requirements.txt
(cd utils/nms_rotated && python setup.py develop)
(cd DOTA_devkit && swig -c++ -python polyiou.i && python setup.py build_ext --inplace)

# 2) Demo entrenamiento
python train.py --weights weights/yolov5n.pt --data data/yolov5obb_demo.yaml \
  --hyp data/hyps/obb/hyp.finetune_dota.yaml --epochs 10 --batch-size 1 --img 1024 --device 0

# 3) Validación + evaluación OBB
python val.py --data data/yolov5obb_demo.yaml --weights runs/train/exp/weights/best.pt \
  --batch-size 1 --img 2048 --task val --device 0 --save-json --name obb_demo
python tools/TestJson2VocClassTxt.py --json_path runs/val/obb_demo/best_obb_predictions.json \
  --save_path runs/val/obb_demo/obb_predictions_Txt
python DOTA_devkit/dota_evaluation_task1.py \
  --detpath 'runs/val/obb_demo/obb_predictions_Txt/Task1_{:s}.txt' \
  --annopath 'dataset/dataset_demo/labelTxt/{:s}.txt' \
  --imagesetfile dataset/dataset_demo/imgnamefile.txt

# 4) Inferencia
python detect.py --weights runs/train/exp/weights/best.pt --source dataset/dataset_demo/images \
  --img 2048 --device 0 --conf-thres 0.25 --iou-thres 0.2 --save-txt --save-conf

# 5) Clúster LSF
bsub < run.bsub
```

## 9. Estado del plan

| Fase | Estado |
|---|---|
| F0 Baseline | En curso (T-01 ✅) |
| F1 Entorno | Pendiente |
| F2 Tests + CI | Pendiente |
| F3 Hardening clúster | Pendiente |
| F4 Correctitud modelo | Pendiente |
| F5 Dataset propio | Pendiente |
