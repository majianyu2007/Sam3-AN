import atexit
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import app as app_module
from services.annotation_manager import AnnotationManager
from services.sam3_service import SAM3Service


# app.py creates a production manager at import time. Prevent the test interpreter's
# exit hook from rewriting repository data; its daemon thread does not save immediately.
atexit.unregister(app_module.shutdown_services)


class AnnotationManagerTests(unittest.TestCase):
    def test_annotations_are_written_immediately_and_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = AnnotationManager(temp_dir, start_autosave=False)
            manager.create_project({
                "id": "project1",
                "name": "测试项目",
                "classes": ["cat"],
                "images": [{
                    "filename": "cat.jpg",
                    "path": "/tmp/cat.jpg",
                    "annotated": False,
                    "annotations": [],
                }],
            })

            annotation = {
                "id": "annotation1",
                "class_name": "cat",
                "bbox": [1, 2, 3, 4],
            }
            registry_before = (Path(temp_dir) / "projects.json").read_bytes()
            manager.save_annotations("project1", 0, [annotation])

            registry_path = Path(temp_dir) / "projects.json"
            registry = json.loads(registry_path.read_text())
            detail = json.loads(
                (Path(temp_dir) / "project1" / "annotations.json").read_text()
            )
            self.assertEqual(registry["schema_version"], 2)
            self.assertNotIn("images", registry["projects"][0])
            self.assertEqual(registry["projects"][0]["image_count"], 1)
            self.assertEqual(registry_path.read_bytes(), registry_before)
            self.assertEqual(
                detail["images"][0]["annotations"][0]["id"],
                "annotation1",
            )
            self.assertEqual(list(Path(temp_dir).rglob("*.tmp")), [])
            self.assertTrue(
                (Path(temp_dir) / "project1" / "annotations.json.bak").is_file()
            )

            reloaded = AnnotationManager(temp_dir, start_autosave=False)
            self.assertEqual(
                reloaded.get_annotations("project1", 0)[0]["id"],
                "annotation1",
            )

    def test_corrupt_registry_recovers_from_valid_backup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            manager = AnnotationManager(data_dir, start_autosave=False)
            manager.create_project({
                "id": "project1",
                "name": "原始名称",
                "images": [],
                "classes": [],
            })
            manager.update_project("project1", {"name": "更新名称"})
            self.assertTrue((data_dir / "projects.json.bak").is_file())
            (data_dir / "projects.json").write_bytes(b"")

            recovered = AnnotationManager(data_dir, start_autosave=False)
            self.assertEqual(recovered.get_project("project1")["name"], "原始名称")
            parsed = json.loads((data_dir / "projects.json").read_text())
            self.assertEqual(parsed["projects"][0]["id"], "project1")

    def test_negative_and_out_of_range_indices_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = AnnotationManager(temp_dir, start_autosave=False)
            manager.create_project({
                "id": "project1",
                "name": "测试项目",
                "images": [{"filename": "one.jpg", "annotations": []}],
                "classes": [],
            })
            with self.assertRaisesRegex(ValueError, "图片索引越界"):
                manager.save_annotations("project1", -1, [])
            with self.assertRaisesRegex(ValueError, "图片索引越界"):
                manager.get_annotations("project1", 1)


class FlaskContractTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.images_dir = self.root / "images"
        self.images_dir.mkdir()
        Image.new("RGB", (16, 12), "white").save(self.images_dir / "cat.png")
        self.manager = AnnotationManager(
            self.root / "data",
            start_autosave=False,
        )
        self.original_manager = app_module.annotation_manager
        app_module.annotation_manager = self.manager
        app_module.app.config.update(TESTING=True)
        self.client = app_module.app.test_client()

    def tearDown(self):
        self.manager.shutdown()
        app_module.annotation_manager = self.original_manager
        self.temp_dir.cleanup()

    def create_project(self):
        response = self.client.post(
            "/api/project/create",
            json={
                "name": "macOS 项目",
                "image_dir": str(self.images_dir),
                "output_dir": str(self.root / "exports"),
                "classes": ["cat", "cat", ""],
            },
        )
        self.assertEqual(response.status_code, 201)
        return response.get_json()["project"]

    def test_project_scan_and_annotation_save_contract(self):
        project = self.create_project()
        self.assertEqual(project["classes"], ["cat"])
        self.assertTrue(Path(project["output_dir"]).is_dir())

        scan = self.client.post(
            f"/api/project/{project['id']}/load_images",
            json={"image_dir": str(self.images_dir)},
        )
        self.assertEqual(scan.status_code, 200)
        self.assertEqual(scan.get_json()["count"], 1)

        saved = self.client.post(
            "/api/annotation/save",
            json={
                "project_id": project["id"],
                "image_index": 0,
                "annotations": [{"id": "a1", "class_name": "cat"}],
            },
        )
        self.assertEqual(saved.status_code, 200)
        detail = json.loads(
            (self.root / "data" / project["id"] / "annotations.json").read_text()
        )
        self.assertEqual(detail["images"][0]["annotations"][0]["id"], "a1")

        project_list = self.client.get("/api/project/list")
        self.assertEqual(project_list.status_code, 200)
        summary = project_list.get_json()["projects"][0]
        self.assertNotIn("images", summary)
        self.assertEqual(summary["image_count"], 1)
        self.assertEqual(summary["annotated_count"], 1)

        invalid = self.client.post(
            "/api/annotation/save",
            json={
                "project_id": project["id"],
                "image_index": -1,
                "annotations": [],
            },
        )
        self.assertEqual(invalid.status_code, 400)
        self.assertFalse(invalid.get_json()["success"])

    def test_windows_path_is_rejected_with_json_error(self):
        response = self.client.post(
            "/api/project/create",
            json={"name": "旧项目", "image_dir": r"C:\\dataset\\images"},
        )
        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertFalse(payload["success"])
        self.assertIn("Windows 路径", payload["error"])

    def test_inference_failure_is_not_reported_as_empty_success(self):
        project = self.create_project()
        self.client.post(
            f"/api/project/{project['id']}/load_images",
            json={"image_dir": str(self.images_dir)},
        )
        fake_service = unittest.mock.Mock()
        fake_service.segment_by_text.side_effect = RuntimeError("MPS 内存不足")
        with patch.object(app_module, "get_sam3_service", return_value=fake_service):
            response = self.client.post(
                "/api/segment/text",
                json={
                    "image_path": str(self.images_dir / "cat.png"),
                    "prompt": "cat",
                    "confidence": 0.5,
                },
            )
        self.assertEqual(response.status_code, 500)
        payload = response.get_json()
        self.assertFalse(payload["success"])
        self.assertIn("MPS 内存不足", payload["error"])

    def test_system_status_and_picker_cancel_contracts(self):
        status = self.client.get("/api/system/status")
        self.assertEqual(status.status_code, 200)
        self.assertIn(status.get_json()["device"], {"cuda", "mps", "cpu"})

        with patch.object(app_module, "choose_native_directory", return_value=None):
            picker = self.client.post(
                "/api/system/select-directory",
                json={"purpose": "image"},
            )
        self.assertEqual(picker.status_code, 200)
        self.assertTrue(picker.get_json()["canceled"])


class ServiceConcurrencyTests(unittest.TestCase):
    def test_lazy_service_is_constructed_once(self):
        original = app_module.sam3_service
        app_module.sam3_service = None
        sentinel = object()
        try:
            with patch.object(app_module, "SAM3Service", return_value=sentinel) as factory:
                with ThreadPoolExecutor(max_workers=8) as executor:
                    services = list(executor.map(lambda _: app_module.get_sam3_service(), range(16)))
            self.assertEqual(factory.call_count, 1)
            self.assertTrue(all(service is sentinel for service in services))
        finally:
            app_module.sam3_service = original

    def test_service_re_raises_model_failures(self):
        service = SAM3Service()
        with patch.object(
            service,
            "_init_image_model",
            side_effect=RuntimeError("模型加载失败"),
        ):
            with self.assertRaisesRegex(RuntimeError, "文本分割失败"):
                service.segment_by_text("unused.png", "cat")


if __name__ == "__main__":
    unittest.main()
