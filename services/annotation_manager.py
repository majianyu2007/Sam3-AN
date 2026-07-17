"""
标注数据管理器。
负责项目、图片、标注的增删改查与安全持久化。
"""

import copy
import os
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path

import orjson


class AnnotationManager:
    """线程安全的标注数据管理器。"""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        autosave_interval: float = 60,
        start_autosave: bool = True,
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
        self.projects = {
            project["id"]: project
            for project in projects
            if isinstance(project, dict) and project.get("id")
        }

    def autosave_loop(self):
        """等待一个周期后再保存；启动时不立即覆盖刚读取的数据。"""
        while not self._stop_event.wait(self._autosave_interval):
            try:
                self.flush()
                print("[AUTOSAVE] SAVED")
            except Exception as error:
                print(f"[ERROR] 自动保存失败: {error}")

    def _save_all_projects(self):
        with self._lock:
            self._atomic_write_json(
                self.data_dir / "projects.json",
                {
                    "projects": list(self.projects.values()),
                    "updated_at": datetime.now().isoformat(),
                },
            )

    def _save_project_annotations(self, project_id: str):
        with self._lock:
            project = self.projects.get(project_id)
            if not project:
                return
            self._atomic_write_json(
                self.data_dir / project_id / "annotations.json",
                {
                    "project_id": project_id,
                    "images": project.get("images", []),
                    "classes": project.get("classes", []),
                    "updated_at": datetime.now().isoformat(),
                },
            )

    def flush(self):
        """同步保存全部状态。"""
        with self._lock:
            self._save_all_projects()
            for project_id in list(self.projects):
                self._save_project_annotations(project_id)

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
            project = copy.deepcopy(project)
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
            self._save_all_projects()
            self._save_project_annotations(project_id)
            return copy.deepcopy(project)

    def get_project(self, project_id: str) -> dict | None:
        with self._lock:
            project = self.projects.get(project_id)
            return copy.deepcopy(project) if project else None

    def list_projects(self) -> list:
        with self._lock:
            return copy.deepcopy(list(self.projects.values()))

    def update_project(self, project_id: str, updates: dict) -> dict:
        with self._lock:
            project = self._require_project(project_id)
            project.update(copy.deepcopy(updates))
            project["updated_at"] = datetime.now().isoformat()
            self._save_all_projects()
            self._save_project_annotations(project_id)
            return copy.deepcopy(project)

    def delete_project(self, project_id: str):
        with self._lock:
            self._require_project(project_id)
            del self.projects[project_id]
            self._save_all_projects()
            project_dir = self.data_dir / project_id
            if project_dir.exists():
                shutil.rmtree(project_dir)

    def update_project_images(self, project_id: str, images: list, image_dir: str):
        with self._lock:
            project = self._require_project(project_id)
            existing_annotations = {
                image["filename"]: copy.deepcopy(image.get("annotations", []))
                for image in project.get("images", [])
                if image.get("annotations")
            }
            images = copy.deepcopy(images)
            for image in images:
                annotations = existing_annotations.get(image["filename"])
                if annotations:
                    image["annotations"] = annotations
                    image["annotated"] = True
            project["images"] = images
            project["image_dir"] = image_dir
            project["current_index"] = min(
                project.get("current_index", 0), max(len(images) - 1, 0)
            )
            project["updated_at"] = datetime.now().isoformat()
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
            annotations = copy.deepcopy(annotations)
            for annotation in annotations:
                if label:
                    annotation["class_name"] = label
                annotation.setdefault("id", str(uuid.uuid4())[:8])
            image.setdefault("annotations", []).extend(annotations)
            image["annotated"] = bool(image["annotations"])
            project["updated_at"] = datetime.now().isoformat()
            self._save_all_projects()
            self._save_project_annotations(project_id)

    def save_annotations(self, project_id: str, image_index: int, annotations: list):
        with self._lock:
            project = self._require_project(project_id)
            image = self._require_image(project, image_index)
            image["annotations"] = copy.deepcopy(annotations)
            image["annotated"] = bool(annotations)
            project["current_index"] = image_index
            project["updated_at"] = datetime.now().isoformat()
            self._save_all_projects()
            self._save_project_annotations(project_id)

    def get_annotations(self, project_id: str, image_index: int) -> list:
        with self._lock:
            project = self._require_project(project_id)
            image = self._require_image(project, image_index)
            return copy.deepcopy(image.get("annotations", []))

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
            annotation = next(
                (
                    item
                    for item in image.get("annotations", [])
                    if item.get("id") == annotation_id
                ),
                None,
            )
            if annotation is None:
                raise ValueError(f"标注不存在: {annotation_id}")
            annotation.update(copy.deepcopy(updates))
            project["updated_at"] = datetime.now().isoformat()
            self._save_all_projects()
            self._save_project_annotations(project_id)

    def delete_annotation(self, project_id: str, image_index: int, annotation_id: str):
        with self._lock:
            project = self._require_project(project_id)
            image = self._require_image(project, image_index)
            annotations = image.get("annotations", [])
            remaining = [item for item in annotations if item.get("id") != annotation_id]
            if len(remaining) == len(annotations):
                raise ValueError(f"标注不存在: {annotation_id}")
            image["annotations"] = remaining
            image["annotated"] = bool(remaining)
            project["updated_at"] = datetime.now().isoformat()
            self._save_all_projects()
            self._save_project_annotations(project_id)

    def update_classes(self, project_id: str, classes: list):
        with self._lock:
            project = self._require_project(project_id)
            project["classes"] = copy.deepcopy(classes)
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
            image["annotated"] = annotated
            project["updated_at"] = datetime.now().isoformat()
            self._save_all_projects()
            self._save_project_annotations(project_id)

    def get_annotation_stats(self, project_id: str) -> dict:
        with self._lock:
            project = self._require_project(project_id)
            images = project.get("images", [])
            total = len(images)
            annotated = sum(1 for image in images if image.get("annotated", False))
            total_annotations = sum(
                len(image.get("annotations", [])) for image in images
            )
            return {
                "total_images": total,
                "annotated_images": annotated,
                "unannotated_images": total - annotated,
                "total_annotations": total_annotations,
                "progress": annotated / total * 100 if total else 0,
            }
