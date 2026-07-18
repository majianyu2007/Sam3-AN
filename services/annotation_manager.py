"""
标注数据管理器。
负责项目、图片、标注的增删改查与安全持久化。
"""

from collections import OrderedDict
import hashlib
import os
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path

import orjson

def _json_clone(value):
    """通过已验证的 JSON 表示创建隔离副本，避免递归 Python deepcopy。"""
    return orjson.loads(orjson.dumps(value))



class AnnotationManager:
    """线程安全的标注数据管理器。"""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        autosave_interval: float = 60,
        start_autosave: bool = True,
        annotation_cache_size: int = 16,
    ):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.projects = {}
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._autosave_interval = autosave_interval
        self._closed = False
        self._sidecars_ready = set()
        self._registry_dirty = False
        self._annotation_cache_size = max(1, int(annotation_cache_size))
        self._annotation_cache = OrderedDict()
        self.thread = None
        self._load_all_projects()

        if start_autosave:
            self.thread = threading.Thread(
                target=self.autosave_loop,
                name="annotation-autosave",
                daemon=True,
            )
            self.thread.start()

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime("%Y%m%d-%H%M%S")

    @staticmethod
    def _read_json_file(path: Path) -> dict:
        payload = path.read_bytes()
        if not payload:
            raise ValueError("文件为空")
        data = orjson.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("JSON 根节点必须是对象")
        return data

    def _atomic_write_json(
        self,
        path: Path,
        data: dict,
        *,
        create_backup: bool = True,
    ):
        """写入临时文件并原子替换，避免进程中断留下空 JSON。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        backup_path = path.with_suffix(path.suffix + ".bak")
        payload = orjson.dumps(data, option=orjson.OPT_INDENT_2)

        try:
            with temp_path.open("wb") as file:
                file.write(payload)
                file.flush()
                os.fsync(file.fileno())

            if create_backup and path.exists():
                try:
                    self._read_json_file(path)
                except (OSError, ValueError, orjson.JSONDecodeError):
                    pass
                else:
                    shutil.copy2(path, backup_path)

            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)

    def _load_json_with_recovery(self, path: Path, default: dict) -> dict:
        if not path.exists():
            return default

        try:
            return self._read_json_file(path)
        except (OSError, ValueError, orjson.JSONDecodeError) as error:
            print(f"[WARN] 无法读取 {path}: {error}")

        backup_path = path.with_suffix(path.suffix + ".bak")
        if backup_path.exists():
            try:
                recovered = self._read_json_file(backup_path)
            except (OSError, ValueError, orjson.JSONDecodeError) as error:
                print(f"[WARN] 备份文件也无法读取 {backup_path}: {error}")
            else:
                print(f"[WARN] 已从备份恢复 {path}")
                self._atomic_write_json(path, recovered, create_backup=False)
                return recovered

        corrupt_path = path.with_name(
            f"{path.stem}.corrupt-{self._timestamp()}{path.suffix}"
        )
        try:
            os.replace(path, corrupt_path)
            print(f"[WARN] 损坏文件已保留为 {corrupt_path}")
        except OSError as error:
            print(f"[WARN] 无法隔离损坏文件 {path}: {error}")
        return default

    def _load_all_projects(self):
        projects_file = self.data_dir / "projects.json"
        data = self._load_json_with_recovery(projects_file, {"projects": []})
        projects = data.get("projects", [])
        if not isinstance(projects, list):
            raise ValueError("projects.json 中的 projects 必须是数组")

        registry_schema = data.get("schema_version", 1)
        loaded = {}
        for stored_project in projects:
            if not isinstance(stored_project, dict) or not stored_project.get("id"):
                continue
            project = _json_clone(stored_project)
            project_id = project["id"]
            detail_path = self.data_dir / project_id / "annotations.json"
            detail = self._load_json_with_recovery(detail_path, {})
            detail_schema = detail.get("schema_version", 1)
            detail_images = detail.get("images")
            detail_updated = detail.get("updated_at", "")
            registry_updated = project.get("updated_at", "")
            use_detail = (
                isinstance(detail_images, list)
                and (
                    registry_schema >= 2
                    or "images" not in project
                    or detail_updated >= registry_updated
                )
            )
            if use_detail:
                project["images"] = detail_images
                if isinstance(detail.get("classes"), list):
                    project["classes"] = detail["classes"]
                if detail_updated > registry_updated:
                    project["updated_at"] = detail_updated
            else:
                project.setdefault("images", [])
            project.setdefault("classes", [])
            project.pop("image_count", None)
            project.pop("annotated_count", None)
            if self._overlay_sidecar_presence(
                project,
                preserve_legacy=detail_schema < 3,
            ):
                self._registry_dirty = True
            if detail_schema >= 3:
                self._sidecars_ready.add(project_id)
            loaded[project_id] = project
        self.projects = loaded
        if registry_schema < 3:
            self._registry_dirty = True

    @staticmethod
    def _image_sidecar_name(filename: str) -> str:
        digest = hashlib.sha256(filename.encode("utf-8")).hexdigest()
        return f"{digest}.json"

    def _image_sidecar_path(self, project_id: str, filename: str) -> Path:
        return (
            self.data_dir
            / project_id
            / "image_annotations"
            / self._image_sidecar_name(filename)
        )

    def _overlay_sidecar_presence(
        self,
        project: dict,
        *,
        preserve_legacy: bool,
    ) -> bool:
        """仅按 sidecar 文件名恢复 annotated 状态，不解析标注正文。"""
        sidecar_dir = self.data_dir / project["id"] / "image_annotations"
        sidecar_names = set()
        if sidecar_dir.is_dir():
            sidecar_names = {
                path.name
                for path in sidecar_dir.iterdir()
                if (
                    path.suffix == ".json"
                    and ".corrupt-" not in path.name
                    and path.is_file()
                )
            }

        changed = False
        for image in project.get("images", []):
            legacy_annotations = image.get("annotations")
            has_sidecar = (
                self._image_sidecar_name(str(image.get("filename", "")))
                in sidecar_names
            )
            annotated = has_sidecar or (
                preserve_legacy
                and isinstance(legacy_annotations, list)
                and bool(legacy_annotations)
            )
            if bool(image.get("annotated")) != annotated:
                changed = True
            image["annotated"] = annotated
            if not preserve_legacy:
                image.pop("annotations", None)
        return changed

    def autosave_loop(self):
        """等待一个周期后再保存；启动时不立即覆盖刚读取的数据。"""
        while not self._stop_event.wait(self._autosave_interval):
            try:
                if self.flush():
                    print("[AUTOSAVE] SAVED")
            except Exception as error:
                print(f"[ERROR] 自动保存失败: {error}")

    @staticmethod
    def _registry_record(project: dict) -> dict:
        """注册表仅保存项目元数据；大体积图片/标注留在项目明细文件。"""
        record = {
            key: value
            for key, value in project.items()
            if key != "images"
        }
        images = project.get("images", [])
        record["image_count"] = len(images)
        record["annotated_count"] = sum(
            1 for image in images if image.get("annotated", False)
        )
        return record


    def _save_all_projects(self):
        with self._lock:
            self._atomic_write_json(
                self.data_dir / "projects.json",
                {
                    "schema_version": 3,
                    "projects": [
                        self._registry_record(project)
                        for project in self.projects.values()
                    ],
                    "updated_at": datetime.now().isoformat(),
                },
            )
            self._registry_dirty = False

    @staticmethod
    def _annotation_cache_key(project_id: str, filename: str) -> tuple:
        return project_id, filename

    def _cache_annotations(
        self,
        project_id: str,
        filename: str,
        annotations: list,
    ) -> list:
        key = self._annotation_cache_key(project_id, filename)
        self._annotation_cache[key] = annotations
        self._annotation_cache.move_to_end(key)
        while len(self._annotation_cache) > self._annotation_cache_size:
            self._annotation_cache.popitem(last=False)
        return annotations

    def _load_image_annotations(
        self,
        project_id: str,
        image: dict,
    ) -> list:
        filename = image.get("filename")
        if not filename:
            raise ValueError("图片缺少文件名，无法读取标注")
        key = self._annotation_cache_key(project_id, filename)
        if key in self._annotation_cache:
            self._annotation_cache.move_to_end(key)
            return self._annotation_cache[key]

        legacy_annotations = image.get("annotations")
        if isinstance(legacy_annotations, list):
            annotations = _json_clone(legacy_annotations)
        else:
            payload = self._load_json_with_recovery(
                self._image_sidecar_path(project_id, filename),
                {},
            )
            if (
                payload.get("project_id") == project_id
                and payload.get("filename") == filename
                and isinstance(payload.get("annotations"), list)
            ):
                annotations = payload["annotations"]
            else:
                annotations = []

        if bool(image.get("annotated")) != bool(annotations):
            image["annotated"] = bool(annotations)
            self._registry_dirty = True
        return self._cache_annotations(project_id, filename, annotations)

    def _save_image_annotations(
        self,
        project_id: str,
        image: dict,
        annotations: list,
    ):
        filename = image.get("filename")
        if not filename:
            raise ValueError("图片缺少文件名，无法保存标注")
        project = self._require_project(project_id)
        annotations = _json_clone(annotations)
        sidecar_path = self._image_sidecar_path(project_id, filename)
        if annotations:
            self._atomic_write_json(
                sidecar_path,
                {
                    "schema_version": 3,
                    "project_id": project_id,
                    "filename": filename,
                    "annotated": True,
                    "annotations": annotations,
                    "updated_at": project.get(
                        "updated_at",
                        datetime.now().isoformat(),
                    ),
                },
            )
        else:
            sidecar_path.unlink(missing_ok=True)
            sidecar_path.with_suffix(sidecar_path.suffix + ".bak").unlink(
                missing_ok=True,
            )
        image.pop("annotations", None)
        image["annotated"] = bool(annotations)
        self._cache_annotations(project_id, filename, annotations)

    def _save_all_image_annotations(self, project_id: str):
        project = self.projects.get(project_id)
        if not project:
            return
        for image in project.get("images", []):
            annotations = image.get("annotations")
            if isinstance(annotations, list):
                self._save_image_annotations(
                    project_id,
                    image,
                    annotations,
                )

    def _save_project_annotations(self, project_id: str):
        with self._lock:
            project = self.projects.get(project_id)
            if not project:
                return
            if project_id not in self._sidecars_ready:
                self._save_all_image_annotations(project_id)
            for image in project.get("images", []):
                image.pop("annotations", None)
            self._atomic_write_json(
                self.data_dir / project_id / "annotations.json",
                {
                    "schema_version": 3,
                    "project_id": project_id,
                    "images": project.get("images", []),
                    "classes": project.get("classes", []),
                    "updated_at": project.get(
                        "updated_at",
                        datetime.now().isoformat(),
                    ),
                },
            )
            self._sidecars_ready.add(project_id)

    def flush(self) -> bool:
        """同步未落盘的轻量元数据，并完成旧格式的一次性迁移。"""
        with self._lock:
            saved = False
            for project_id in list(self.projects):
                if project_id not in self._sidecars_ready:
                    self._save_project_annotations(project_id)
                    saved = True
            if self._registry_dirty:
                self._save_all_projects()
                saved = True
            return saved

    def shutdown(self):
        """停止自动保存线程并完成最后一次同步保存。可重复调用。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._stop_event.set()
        if (
            self.thread
            and self.thread.is_alive()
            and threading.current_thread() is not self.thread
        ):
            self.thread.join(timeout=min(self._autosave_interval, 2))
        self.flush()

    def _require_project(self, project_id: str) -> dict:
        project = self.projects.get(project_id)
        if not project:
            raise ValueError(f"项目不存在: {project_id}")
        return project

    @staticmethod
    def _require_image(project: dict, image_index: int) -> dict:
        if not isinstance(image_index, int):
            raise ValueError("图片索引必须是整数")
        images = project.get("images", [])
        if image_index < 0 or image_index >= len(images):
            raise ValueError(f"图片索引越界: {image_index}")
        return images[image_index]

    def create_project(self, project: dict) -> dict:
        with self._lock:
            project = _json_clone(project)
            project_id = project.get("id", str(uuid.uuid4())[:8])
            if project_id in self.projects:
                raise ValueError(f"项目已存在: {project_id}")
            project["id"] = project_id
            project.setdefault("created_at", datetime.now().isoformat())
            project["updated_at"] = datetime.now().isoformat()
            project.setdefault("images", [])
            project.setdefault("classes", [])
            self.projects[project_id] = project
            (self.data_dir / project_id).mkdir(parents=True, exist_ok=True)
            self._save_project_annotations(project_id)
            self._save_all_projects()
            return self._hydrate_project(project)

    @staticmethod
    def _project_manifest(project: dict) -> dict:
        """返回不含标注大数组的项目导航清单。"""
        manifest = {
            key: value
            for key, value in project.items()
            if key != "images"
        }
        manifest["images"] = [
            {
                key: value
                for key, value in image.items()
                if key != "annotations"
            }
            for image in project.get("images", [])
        ]
        return manifest

    def _hydrate_project(self, project: dict) -> dict:
        hydrated = _json_clone(self._project_manifest(project))
        for index, image in enumerate(project.get("images", [])):
            hydrated["images"][index]["annotations"] = _json_clone(self._load_image_annotations(project["id"], image))
        return hydrated

    def get_project_manifest(self, project_id: str) -> dict | None:
        with self._lock:
            project = self.projects.get(project_id)
            return _json_clone(self._project_manifest(project)) if project else None

    def get_project(self, project_id: str) -> dict | None:
        """兼容完整项目读取；标注逐图加载且不驻留在项目 manifest。"""
        with self._lock:
            project = self.projects.get(project_id)
            return self._hydrate_project(project) if project else None

    def list_projects(self) -> list:
        with self._lock:
            return [
                self._hydrate_project(project)
                for project in self.projects.values()
            ]
    def list_project_summaries(self) -> list:
        """返回不含图片和标注大数组的轻量项目列表。"""
        with self._lock:
            return _json_clone([
                self._registry_record(project)
                for project in self.projects.values()
            ])


    def update_project(self, project_id: str, updates: dict) -> dict:
        with self._lock:
            project = self._require_project(project_id)
            project.update(_json_clone(updates))
            project["updated_at"] = datetime.now().isoformat()
            self._save_all_projects()
            self._save_project_annotations(project_id)
            return _json_clone(project)

    def delete_project(self, project_id: str):
        with self._lock:
            self._require_project(project_id)
            del self.projects[project_id]
            self._save_all_projects()
            project_dir = self.data_dir / project_id
            if project_dir.exists():
                shutil.rmtree(project_dir)
            stale_keys = [
                key for key in self._annotation_cache
                if key[0] == project_id
            ]
            for key in stale_keys:
                self._annotation_cache.pop(key, None)

    def update_project_images(self, project_id: str, images: list, image_dir: str):
        with self._lock:
            project = self._require_project(project_id)
            if project_id not in self._sidecars_ready:
                self._save_project_annotations(project_id)
            existing_annotated = {
                image.get("filename")
                for image in project.get("images", [])
                if image.get("annotated")
            }
            images = _json_clone(images)
            new_annotations = []
            for image in images:
                filename = image.get("filename")
                if not filename:
                    raise ValueError("图片缺少文件名")
                annotations = image.pop("annotations", [])
                has_existing = (
                    filename in existing_annotated
                    or self._image_sidecar_path(project_id, filename).is_file()
                )
                image["annotated"] = has_existing or bool(annotations)
                if annotations and not has_existing:
                    new_annotations.append((image, annotations))
            project["images"] = images
            project["image_dir"] = image_dir
            project["current_index"] = min(
                project.get("current_index", 0),
                max(len(images) - 1, 0),
            )
            project["updated_at"] = datetime.now().isoformat()
            for image, annotations in new_annotations:
                self._save_image_annotations(
                    project_id,
                    image,
                    annotations,
                )
            self._save_all_projects()
            self._save_project_annotations(project_id)

    def add_annotations(
        self,
        project_id: str,
        image_index: int,
        annotations: list,
        label: str | None = None,
    ):
        with self._lock:
            project = self._require_project(project_id)
            image = self._require_image(project, image_index)
            combined = _json_clone(self._load_image_annotations(project_id, image))
            additions = _json_clone(annotations)
            for annotation in additions:
                if label:
                    annotation["class_name"] = label
                annotation.setdefault("id", str(uuid.uuid4())[:8])
            combined.extend(additions)
            project["updated_at"] = datetime.now().isoformat()
            self._save_image_annotations(project_id, image, combined)
            self._registry_dirty = True

    def save_annotations(self, project_id: str, image_index: int, annotations: list):
        with self._lock:
            project = self._require_project(project_id)
            image = self._require_image(project, image_index)
            project["current_index"] = image_index
            project["updated_at"] = datetime.now().isoformat()
            self._save_image_annotations(project_id, image, annotations)
            self._registry_dirty = True

    def get_annotations(self, project_id: str, image_index: int) -> list:
        with self._lock:
            project = self._require_project(project_id)
            image = self._require_image(project, image_index)
            return _json_clone(self._load_image_annotations(project_id, image))
    def get_project_image(
        self,
        project_id: str,
        image_index: int,
        *,
        include_annotations: bool = True,
    ) -> dict:
        with self._lock:
            project = self._require_project(project_id)
            image = self._require_image(project, image_index)
            result = _json_clone(image)
            if include_annotations:
                result["annotations"] = _json_clone(self._load_image_annotations(project_id, image))
            return result

    def update_annotation(
        self,
        project_id: str,
        image_index: int,
        annotation_id: str,
        updates: dict,
    ):
        with self._lock:
            project = self._require_project(project_id)
            image = self._require_image(project, image_index)
            annotations = _json_clone(self._load_image_annotations(project_id, image))
            annotation = next(
                (
                    item
                    for item in annotations
                    if item.get("id") == annotation_id
                ),
                None,
            )
            if annotation is None:
                raise ValueError(f"标注不存在: {annotation_id}")
            annotation.update(_json_clone(updates))
            project["updated_at"] = datetime.now().isoformat()
            self._save_image_annotations(project_id, image, annotations)
            self._registry_dirty = True

    def delete_annotation(self, project_id: str, image_index: int, annotation_id: str):
        with self._lock:
            project = self._require_project(project_id)
            image = self._require_image(project, image_index)
            annotations = self._load_image_annotations(project_id, image)
            remaining = [
                item for item in annotations
                if item.get("id") != annotation_id
            ]
            if len(remaining) == len(annotations):
                raise ValueError(f"标注不存在: {annotation_id}")
            project["updated_at"] = datetime.now().isoformat()
            self._save_image_annotations(project_id, image, remaining)
            self._registry_dirty = True

    def update_classes(self, project_id: str, classes: list):
        with self._lock:
            project = self._require_project(project_id)
            project["classes"] = _json_clone(classes)
            project["updated_at"] = datetime.now().isoformat()
            self._save_all_projects()
            self._save_project_annotations(project_id)

    def mark_image_annotated(
        self,
        project_id: str,
        image_index: int,
        annotated: bool = True,
    ):
        with self._lock:
            project = self._require_project(project_id)
            image = self._require_image(project, image_index)
            annotations = self._load_image_annotations(project_id, image)
            if annotated and not annotations:
                raise ValueError("空标注图片不能标记为已标注")
            if not annotated:
                annotations = []
            project["updated_at"] = datetime.now().isoformat()
            self._save_image_annotations(project_id, image, annotations)
            self._registry_dirty = True

    def get_annotation_stats(self, project_id: str) -> dict:
        with self._lock:
            project = self._require_project(project_id)
            images = project.get("images", [])
            total = len(images)
            annotated = sum(1 for image in images if image.get("annotated", False))
            total_annotations = sum(
                len(self._load_image_annotations(project_id, image))
                for image in images
            )
            return {
                "total_images": total,
                "annotated_images": annotated,
                "unannotated_images": total - annotated,
                "total_annotations": total_annotations,
                "progress": annotated / total * 100 if total else 0,
            }
