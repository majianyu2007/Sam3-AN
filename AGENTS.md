# Repository Guidelines

> Guidelines for AI assistants working in the **Sam3-AN** codebase. Synthesized from source
> scan of `app.py`, `services/`, `exports/`, `SAM_src/`, `templates/`, `static/`, config files.

## Project Overview

Sam3-AN is a Flask-based **image (and video) annotation tool** built on Meta's SAM3
(Segment Anything Model 3). It turns text/point/box prompts into segmentation masks for generating
YOLO and COCO training datasets. Architecture is a classic 3-tier: Canvas frontend → Flask REST API →
vendored SAM3 PyTorch models. Single-user, local-first (binds `0.0.0.0:5000`, auto-opens browser).

- **Stack**: Python 3.11, Flask 2.3+, PyTorch 2.13 (MPS on macOS / CUDA 12.6 on Linux), Pillow, OpenCV,
  numpy 1.26, timm, decord/eva-decord, pycocotools, orjson.
- **License**: MIT. SAM3 vendored under `SAM_src/` (Meta upstream, v0.1.0).

## Architecture & Data Flow

```
Browser (templates/*.html + static/js/annotation.js, Canvas 2D)
   │  fetch() → /api/*  (JSON envelope {success, ...}*)
Flask (app.py) — ~30 routes across 12 sections
   │  module-level singletons: AnnotationManager (eager), SAM3Service (lazy)
   │  exporters instantiated per-request
services/        exports/
   │                │
   ├─ AnnotationManager  →  data/projects.json + data/<id>/annotations.json
   │                        (threading.Lock, daemon autosave every 60s, orjson)
   └─ SAM3Service  →  SAM_src/sam3 (sys.path.insert at import)
                        build_sam3_image_model(checkpoint_path='sam3.pt')
                        weights: sam3.pt at CWD (HF fallback: facebook/sam3)
```

Key invariants:
- **All endpoints** return `{'success': bool, ...}` (errors carry `'error': str`). Two exceptions noted in `app.py`.
- **SAM3 model is lazy-loaded** once via `get_sam3_service()`; `sys.path.insert(0, "SAM_src")` runs at import time of `app.py` (lines 8–10).
- **State is on-disk JSON**, not a DB. `data/projects.json` is the registry; `data/<project_id>/` holds per-project annotations/images.
- **Threaded server** (`app.run(threaded=True, debug=False)`). `AnnotationManager` guards mutations with `threading.Lock`; **`SAM3Service` lazy init is not locked** — potential race on first request.

## Key Directories

| Path | Purpose |
|------|---------|
| `app.py` | Flask entry; all routes, launch + auto-open-browser routine (`open_browser`/`wait_for_server`). |
| `services/sam3_service.py` | `SAM3Service`: wraps image + video SAM3 predictors; `segment_by_text/points/boxes`, video session methods. |
| `services/annotation_manager.py` | `AnnotationManager`: project/annotation/image CRUD, `threading.Lock`, 60s daemon autosave, orjson. |
| `exports/yolo_exporter.py` | `YOLOExporter.export()`: train/val/test 8:1:1 split, detect|segment, polygon smoothing. |
| `exports/coco_exporter.py` | `COCOExporter.export()`: COCO JSON, detect|segment, polygon smoothing. |
| `templates/index.html` | Image annotation UI (4-panel: sidebar / canvas / annotation list / class panel). Loads `annotation.js`. |
| `templates/video.html` | Video annotation UI (inline JS; **fully wired despite README's "暂不支持" note**). Calls `/api/video/*`. |
| `static/js/annotation.js` | Single 84 KB unbundled vanilla JS; `const state = {...}` module pattern; Canvas 2D + offscreen cache; all I/O via `fetch('/api/*')`. |
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

# Model weight: download sam3.pt from https://www.modelscope.cn/models/facebook/sam3
#   and place at project root (CWD). HF fallback downloads from facebook/sam3 on first run.

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
- **Persistence via orjson + threading.Lock**: `AnnotationManager` serializes to `data/` under a lock with a 60s
  daemon autosave thread. New stateful managers should follow the same lock + autosave pattern.
- **Routing layout**: `app.py` is organized by `# ==================== <Section> API ====================` comment banners
  (Project / Image / SAM3 segment / Annotation / Classes / Export / Export preview / Video / AI translate / App exit).
  Add new routes under the matching banner.
- **Exporters**: per-request instantiation; both `YOLOExporter` and `COCOExporter` expose `.export(...)` with
  detect|segment mode and polygon smoothing. Pipeline: handlers build args → exporter.export() → return saved paths.
- **Naming**: snake_case throughout Python; Flask routes use `/api/<resource>/<action>` (e.g. `/api/project/<id>/update`).
- **i18n**: docstrings and UI strings are **Chinese**; user-facing error messages are Chinese. Preserve when editing.
- **Frontend**: vanilla JS in `static/js/annotation.js`, no framework/bundler. Add features by extending the `state`
  object and calling existing `fetch('/api/...')` helpers. Video page uses its own inline JS in `templates/video.html`.
- **SAM3 source path hack**: `app.py` does `sys.path.insert(0, "SAM_src")` before importing SAM3 symbols. Any module needing SAM3 must rely on this side effect (or re-insert) — do not `pip install` the vendored package.
- **Device selection (cross-platform)**: `services/sam3_service.py:_select_device()` picks CUDA > MPS (macOS Apple Silicon) > CPU at image-model init. The SAM3 factory `build_sam3_image_model` only moves the model when `device=="cuda"`; for MPS we build on CPU then `.to("mps")`. Force a device via `SAM3_DEVICE=cuda|mps|cpu` (falls back to CPU if unavailable). `Sam3Processor(device=...)` must be passed explicitly — its default is `"cuda"`.
- **Vendored patch — `SAM_src/sam3/model/edt.py`**: `triton` has no macOS wheels, so `import triton` at module top would crash the whole `sam3` import chain (image path included) on macOS. We guarded the import with a stub (`_HAS_TRITON`) that lets `import sam3` succeed; the stub's `edt_triton` only raises if actually called (it asserts `data.is_cuda`, so it never runs on MPS/CPU anyway). Keep this guard when updating vendored SAM3.

## Important Files

- **Entry point**: `app.py` (lines 1–11 bootstrapping; 965–979 launch block).
- **Config**: `pyproject.toml` (deps, `requires-python = ">=3.11,<3.13"`), `uv.lock` (resolved versions),
  `.python-version` (uv Python pin), `requirements.txt` (legacy source list with platform markers), `.gitignore`.
  No settings module — env var `SAM3_DEVICE` (`cuda|mps|cpu`) can override the inference device in `sam3_service.py`.
  AI-translate API config (key/url/model) is stored per-project in `data/`.
- **Model factory**: `SAM_src/sam3/model_builder.py:560-561,640-646` — default `checkpoint_path='sam3.pt'`;
  HF fallback via `huggingface_hub.hf_hub_download('facebook/sam3', 'sam3.pt')`.
- **State files**: `data/projects.json` (registry), `data/<project_id>/annotations.json` (per-project annotations).
- **⚠ Autosave race**: `_save_all_projects` (`services/annotation_manager.py`) opens `projects.json` in `'wb'` (truncates) then writes, non-atomically. If the process dies mid-write the file is left empty → next start fails with `orjson.JSONDecodeError` on a zero-length doc. Avoid abrupt exits while annotation_manager is loaded; consider atomic write (tmp + os.replace) when touching this code.
- **Vendor metadata**: `SAM_src/sam3.egg-info/PKG-INFO` (sam3 v0.1.0, requires-python >=3.8).

## Runtime / Tooling Preferences

- **Runtime**: CPython 3.11, pinned via `.python-version` (uv-managed). `requires-python = ">=3.11,<3.13"`
  — the `<3.13` upper bound respects `numpy==1.26` (broken on 3.13+). README's "3.10+" predates the uv migration.
- **Server**: `app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)`. The background
  `open_browser` thread polls `wait_for_server` then launches the default browser at `http://localhost:5000`.
- **GPU/device**: `_select_device()` → CUDA > MPS > CPU. macOS Apple Silicon runs the **image model on MPS** (no CUDA). Force CPU if you hit MPS op errors: `SAM3_DEVICE=cpu uv run python app.py`. CUDA Linux servers use the `cu126` index (`--index-url https://download.pytorch.org/whl/cu126`) for torch. CPU works but is slow. ~6–8 GB VRAM recommended.
- **Browser launch**: `open_browser` in `app.py` detects installed Chromium browsers per platform (Windows exe paths; macOS `/Applications/*.app/Contents/MacOS/*`; Linux falls back to `webbrowser.open`) and opens in `--app` mode.
- **Package manager**: **uv** with lockfile `uv.lock`. No pip/Pipfile/poetry.
- **No linter/formatter config at root**. When editing SAM3 vendor code, follow its black/ruff/usort conventions.

## Testing & QA

- **No project-level test suite.** The only pytest file is `SAM_src/sam3/perflib/tests/tests.py`
  (single class `TestMasksToBoxes`, upstream vendor test). A standalone `test_reindex_function()` exists in
  `SAM_src/sam3/eval/coco_reindex.py` under `__main__` (not pytest-collected).
- **No CI config** (no `.github/`, no `tox.ini`, no `pytest.ini`).
- **Verification practice**: smoke-test changes by running `python app.py` and exercising the affected
  `/api/*` route through the browser UI or a `curl` POST. Confirm the response envelope and that
  `data/projects.json` / per-project annotation files update as expected.
- **Coverage**: not measured. Treat the Flask route layer and `AnnotationManager` persistence as the
  load-bearing surface to manually verify on any change.