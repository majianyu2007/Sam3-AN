# Repository Guidelines

> Guidelines for AI assistants working in the **Sam3-AN** codebase. Synthesized from source
> scan of `app.py`, `services/`, `exports/`, `SAM_src/`, `templates/`, `static/`, config files.

## Project Overview

Sam3-AN is a Flask-based **image annotation tool** built on Meta's SAM3
(Segment Anything Model 3). It turns text/point/box prompts into segmentation masks for generating
YOLO and COCO training datasets. Architecture is a classic 3-tier: Canvas frontend → Flask REST API →
vendored SAM3 PyTorch models. Single-user, local-first (defaults to `127.0.0.1:5000`, auto-opens browser).

- **Stack**: Python 3.11, Flask 2.3+, PyTorch 2.13 (MPS on macOS / CUDA 12.6 on Linux), Pillow, OpenCV,
  numpy 1.26, timm, decord/eva-decord, pycocotools, orjson.
- **License**: MIT. SAM3 vendored under `SAM_src/` (Meta upstream, v0.1.0).

## Architecture & Data Flow

```
Browser (templates/*.html + static/js/annotation.js, Canvas 2D)
   │  fetch() → /api/*  (JSON normally; export-preview success is JPEG)
Flask (app.py) — ~30 routes across 12 sections
   │  module-level singletons: AnnotationManager (eager), SAM3Service (lazy)
   │  exporters instantiated per-request
services/        exports/
   │                │
   ├─ AnnotationManager  →  data/projects.json + data/<id>/annotations.json
   │                        + data/<id>/image_annotations/<sha256>.json
   │                        (schema-v3 manifests, per-image hydration, RLock, orjson)
   └─ SAM3Service  →  SAM_src/sam3 (sys.path.insert at import)
                        build_sam3_image_model(checkpoint_path='sam3.pt')
                        weights: sam3.pt at CWD (HF fallback: facebook/sam3)
```

Key invariants:
- **API contract**: normal endpoints return `{'success': bool, ...}`; errors always use
  `{'success': False, 'error': str}`. Export-preview success responses are binary JPEG with percent-encoded
  JSON metadata in `X-SAM3-Preview-Stats`; preview errors keep the normal JSON envelope.
- **SAM3 model is lazy-loaded** once under `sam3_service_lock` via `get_sam3_service()`;
  `sys.path.insert(0, "SAM_src")` runs at import time.
- **State is on-disk JSON**, not a DB. `data/projects.json` is the registry; `data/<project_id>/` holds per-project annotations/images.
- **Threaded server** (`app.run(threaded=True, debug=False)`). `AnnotationManager` guards mutations with
  `threading.RLock`; `SAM3Service` serializes model inference through its shared inference lock.

## Key Directories

| Path | Purpose |
|------|---------|
| `app.py` | Flask entry; all routes, launch + auto-open-browser routine (`open_browser`/`wait_for_server`). |
| `services/sam3_service.py` | `SAM3Service`: wraps image + video SAM3 predictors; `segment_by_text/points/boxes`, video session methods. |
| `services/annotation_manager.py` | `AnnotationManager`: schema-v3 project manifests, lazy per-image annotation sidecars, `threading.RLock`, autosave, orjson. |
| `exports/yolo_exporter.py` | `YOLOExporter.export()`: train/val/test 8:1:1 split, detect|segment, polygon smoothing. |
| `exports/coco_exporter.py` | `COCOExporter.export()`: COCO JSON, detect|segment, polygon smoothing. |
| `templates/index.html` | Image annotation UI (4-panel: sidebar / canvas / annotation list / class panel). Loads `annotation.js`. |
| `templates/video.html` | Experimental video session UI with placeholder frame/playback/export behavior; disabled unless `SAM3_ENABLE_EXPERIMENTAL_VIDEO=1`. |
| `static/js/annotation.js` | ~100 KB unbundled vanilla JS; Canvas 2D + offscreen cache; JSON calls use `apiRequest()`, binary previews use `imageRequest()`. |
| `static/css/style.css` | Single 43 KB stylesheet (cyberpunk theme). |
| `SAM_src/sam3/` | Vendored SAM3 package (`build_sam3_image_model`, `__version__="0.1.0"`). Submodules: `model/` (`sam3_image.py`, `sam3_image_processor.py`, `sam3_video_predictor.py`, `sam3_tracking_predictor.py`, `encoder.py`, `decoder.py`, `geometry_encoders.py`, `text_encoder_ve.py`), `sam/` (SAM1/2 compat: `MaskDecoder`, `PromptEncoder`, `TwoWayTransformer`). |
| `SAM_src/scripts/` | Upstream eval utilities (`extract_odinw_results.py`, `extract_roboflow_vl100_results.py`, `eval/standalone_cgf1.py`). Not used by the app. |
| `pyproject.toml` | uv project manifest: `requires-python = ">=3.11,<3.13"`, `[project.dependencies]` is the source of truth for deps. |
| `uv.lock` | uv resolved lockfile (numpy 1.26, torch 2.13, decord/eva-decord with platform markers). Commit this. |
| `.python-version` | uv Python pin (`3.11`). |
| `requirements.txt` | Legacy pip-style dep list, retained with platform markers (`decord`/`eva-decord`, `triton-windows win32-only`). Not the source of truth — `pyproject.toml` is. |
| `data/` | Persistent project state — **NOT gitignored** (projects.json committed). |
| `uploads/` | Temp uploads — gitignored. |
| `utils/` | Empty placeholder (`__init__.py` only). |

## Development Commands

```bash
# uv-managed. Python pinned to 3.11 via .python-version (numpy==1.26 caps Python ≤3.12).
uv sync              # create .venv + install from uv.lock (uses pinned 3.11)
uv run python app.py # launch (binds 0.0.0.0:5000, auto-opens browser; threaded, debug=False)

# Dependencies live in pyproject.toml [project.dependencies] (migrated from requirements.txt via
# `uv add -r requirements.txt`). Edit pyproject.toml going forward, not requirements.txt.
#   uv add <pkg>        # add a dependency (+ re-lock + install)
#   uv lock             # re-resolve after editing deps
#   uv run <cmd>        # run anything inside the venv

# Platform markers handle wheels missing for macOS arm64:
#   decord         -> no macOS arm64 wheel; `eva-decord` (drop-in fork, keeps `import decord`) used on darwin
#   triton-windows -> Windows-only; skipped elsewhere via `; sys_platform == "win32"`

# Model weight: `sam3.pt` at project root, 3,450,062,241 bytes.
# Verified mirror SHA-256: 9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e

# CUDA Linux servers: override the torch index for cu126 wheels
#   uv add torch torchvision --index-url https://download.pytorch.org/whl/cu126
#   (macOS uses the default-index torch build, which is MPS/CPU — CUDA not applicable)
```

There is **no build step, no Makefile/Dockerfile** at repo root. The project is now **uv-managed**
(`pyproject.toml` + `uv.lock` + `.python-version`). SAM3 vendor declares `black`/`ruff`/`usort`/`pytest`
only in its own dev extras — not wired for the app.

## Code Conventions & Common Patterns

- **Response envelope**: every Flask handler ends with `jsonify({'success': True, ...})` or
  `jsonify({'success': False, 'error': str(e)})` inside a `try/except`. Match this for any new endpoint.
- **Lazy service loading**: services needing the GPU model go through `get_sam3_service()` (singleton, first-call init).
  Do not import-time initialize SAM3 — it's ~3.2 GB and slow.
- **Persistence via orjson + `threading.RLock`**: `AnnotationManager` uses schema v3:
  `data/projects.json` is a metadata-only registry, `data/<id>/annotations.json` is the image/class
  manifest, and each image's annotations live in `data/<id>/image_annotations/<sha256(filename)>.json`.
  An annotation mutation atomically rewrites only its small image sidecar; dirty registry metadata flushes
  on the delayed 60s autosave or graceful `shutdown()`. Fsynced temp files, `os.replace`, `.bak` recovery,
  and automatic schema-v2 migration apply to every layer.
- **Routing layout**: `app.py` is organized by `# ==================== <Section> API ====================` comment banners
  (Project / Image / SAM3 segment / Annotation / Classes / Export / Export preview / Video / AI translate / App exit).
  Add new routes under the matching banner.
- **Exporters**: per-request instantiation; both expose `.export(...)` with detect|segment mode and polygon
  smoothing. They build in same-filesystem temporary directories, then replace only their owned output paths.
  Re-exports remove stale files; pre-publish failures preserve the last complete dataset. Keep class union,
  coordinate clamping, accurate exported-image counts, and COCO polygon-area semantics aligned.
- **Naming**: snake_case throughout Python; Flask routes use `/api/<resource>/<action>` (e.g. `/api/project/<id>/update`).
- **i18n**: docstrings and UI strings are **Chinese**; user-facing error messages are Chinese. Preserve when editing.
- **Frontend**: vanilla JS in `static/js/annotation.js`, no framework/bundler. Extend `state`; use
  `apiRequest()` for JSON and `imageRequest()` for binary previews. Revoke old Blob URLs when replacing
  previews. Project routes return image manifests without annotation arrays; `selectProject()`/`loadImage()`
  fetch `/api/annotation/get` in parallel with the bitmap and evict the previous image's annotation cache.
  Annotation saves are serialized and revision-checked; mutations must call `recordAnnotationMutation()` to
  update bounded undo history, dirty state, and autosave. Canvas bitmaps stay at source resolution while CSS
  applies zoom. The image sidebar is window-rendered, so never restore full-list DOM rebuilding. AI keys
  belong in `sessionStorage`, never persistent `localStorage`. Video remains separate inline JS in
  `templates/video.html`, is incomplete, and must remain gated by `SAM3_ENABLE_EXPERIMENTAL_VIDEO`.
- **SAM3 source path hack**: `app.py` does `sys.path.insert(0, "SAM_src")` before importing SAM3 symbols. Any module needing SAM3 must rely on this side effect (or re-insert) — do not `pip install` the vendored package.
- **Device selection (cross-platform)**: `services/sam3_service.py:_select_device()` picks CUDA > MPS (macOS Apple Silicon) > CPU at image-model init. `SAM_src/sam3/model_builder.py:_setup_device_and_mode()` uses `model.to(device)` for all devices. Force via `SAM3_DEVICE=cuda|mps|cpu` (unavailable requested devices fall back to CPU). Always construct `Sam3Processor(..., device=device)` explicitly — its upstream default is `"cuda"`.
- **Vendored macOS patches — preserve on SAM3 updates**: `model/edt.py` guards unavailable macOS `triton`; `model_builder.py` uses device-agnostic `.to(device)`; `model/position_encoding.py` and `model/decoder.py` precompute caches on CPU then migrate once to the input device; `sam/transformer.py` invalidates RoPE cache on device mismatch; `model/geometry_encoders.py` avoids CUDA-only pinned-memory transfer semantics; `model/sam3_video_inference.py` disables CUDA autocast decorators when CUDA is unavailable. These are required for real MPS inference or clean macOS startup, not cosmetic changes.

## Important Files

- **Entry point**: `app.py` (lines 1–11 bootstrapping; 965–979 launch block).
- **Config**: `pyproject.toml` (deps, `requires-python = ">=3.11,<3.13"`), `uv.lock` (resolved versions),
  `.python-version` (uv Python pin), `requirements.txt` (legacy source list with platform markers), `.gitignore`.
  No settings module — env var `SAM3_DEVICE` (`cuda|mps|cpu`) can override the inference device in `sam3_service.py`.
  AI translation URL/model/enabled state is kept in browser `localStorage`; its API key is session-only
  (`sessionStorage`) and HTTPS requests retain certificate verification.
- **Model factory/checkpoint**: `SAM_src/sam3/model_builder.py` defaults to `checkpoint_path='sam3.pt'` at CWD. Verified checkpoint: 3,450,062,241 bytes, SHA-256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`; downloaded from public mirror `1038lab/sam3` because official `facebook/sam3` is gated. `*.pt` is gitignored.
- **State files**: `data/projects.json` (registry), `data/<project_id>/annotations.json` (image/class manifest),
  and `data/<project_id>/image_annotations/<sha256(filename)>.json` (per-image annotations).
  Runtime `.bak`, `.corrupt-*.json`, and temp files are gitignored. Annotation mutations persist their
  sidecar immediately; startup overlays sidecars onto the manifest, recovers valid backups, and preserves
  unrecoverable input under timestamped corrupt filenames.
- **Vendor metadata**: `SAM_src/sam3.egg-info/PKG-INFO` (sam3 v0.1.0, requires-python >=3.8).

## Runtime / Tooling Preferences

- **Runtime**: CPython 3.11, pinned via `.python-version` (uv-managed). `requires-python = ">=3.11,<3.13"`
  — the `<3.13` upper bound respects `numpy==1.26` (broken on 3.13+). README's "3.10+" predates the uv migration.
- **Server**: `app.run(host='0.0.0.0', port=<selected>, threaded=True, debug=False)`. `_find_available_port()` tries 5000 → 5001 → 5055 → 8000 → 8080 → OS-assigned. This handles macOS AirPlay Receiver occupying 5000. `open_browser` receives the selected URL.
- **GPU/device**: `_select_device()` → CUDA > MPS > CPU. Strict MPS (no CPU fallback) is verified on Apple M5 / torch 2.13 for text, point, and box prompts. Force CPU: `SAM3_DEVICE=cpu uv run python app.py`. CUDA Linux servers use the `cu126` index. CPU works but is slow.
- **Browser launch**: `open_browser` in `app.py` detects installed Chromium browsers per platform (Windows exe paths; macOS `/Applications/*.app/Contents/MacOS/*`; Linux falls back to `webbrowser.open`) and opens in `--app` mode.
- **Package manager**: **uv** with lockfile `uv.lock`. No pip/Pipfile/poetry.
- **No linter/formatter config at root**. When editing SAM3 vendor code, follow its black/ruff/usort conventions.

## Testing & QA

- **Project stability suite**: `tests/test_stability.py` uses stdlib `unittest` for schema migration,
  lazy per-image sidecar writes/recovery, manifest hydration, route/path/JSON/payload validation, binary
  preview contracts, experimental-video gating/limits, TLS-preserving AI requests, staged deterministic
  YOLO/COCO exports, COCO area/count correctness, singleton concurrency, and inference-error propagation.
  Run: `uv run python -m unittest discover -s tests -p 'test_stability.py' -v`.
- **Vendor tests**: `SAM_src/sam3/perflib/tests/tests.py` contains the upstream `TestMasksToBoxes`; a standalone
  `test_reindex_function()` exists in `SAM_src/sam3/eval/coco_reindex.py`.
- **No CI config** (no `.github/`, no `tox.ini`, no `pytest.ini`).
- **Verification practice**: smoke-test changes by running `uv run python app.py` and exercising the affected
  `/api/*` route through the browser UI or a `curl` POST. Confirm the response envelope and that the registry,
  project manifest, and affected per-image sidecar update as expected.
- **Verified browser path (current)**: an isolated POSIX-path project loaded annotation manifests on demand,
  navigated between annotated/unannotated images and rehydrated correctly, displayed a streamed JPEG export
  preview through a Blob URL, exported a real YOLO image/label pair, restored state after reload without page
  errors, and kept the AI key out of persistent `localStorage`.
- **Coverage**: not measured. Treat Flask route contracts, browser workflows, and `AnnotationManager` persistence as load-bearing verification surfaces.