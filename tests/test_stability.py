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
from exports.coco_exporter import COCOExporter
from exports.yolo_exporter import YOLOExporter


# app.py creates a production manager at import time. Prevent the test interpreter's
# exit hook from rewriting repository data; its daemon thread does not save immediately.
atexit.unregister(app_module.shutdown_services)


class AnnotationManagerTests(unittest.TestCase):
    def test_annotations_write_only_the_changed_image_sidecar(self):
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
            registry_path = Path(temp_dir) / "projects.json"
            detail_path = Path(temp_dir) / "project1" / "annotations.json"
            registry_before = registry_path.read_bytes()
            detail_before = detail_path.read_bytes()
            manager.save_annotations("project1", 0, [annotation])

            registry = json.loads(registry_path.read_text())
            detail = json.loads(detail_path.read_text())
            sidecars = list(
                (Path(temp_dir) / "project1" / "image_annotations").glob("*.json")
            )
            self.assertEqual(registry["schema_version"], 3)
            self.assertNotIn("images", registry["projects"][0])
            self.assertEqual(registry["projects"][0]["image_count"], 1)
            self.assertEqual(registry_path.read_bytes(), registry_before)
            self.assertEqual(detail_path.read_bytes(), detail_before)
            self.assertNotIn("annotations", detail["images"][0])
            self.assertEqual(len(sidecars), 1)
            sidecar = json.loads(sidecars[0].read_text())
            self.assertEqual(sidecar["annotations"][0]["id"], "annotation1")
            self.assertEqual(list(Path(temp_dir).rglob("*.tmp")), [])

            reloaded = AnnotationManager(temp_dir, start_autosave=False)
            self.assertEqual(
                reloaded.get_annotations("project1", 0)[0]["id"],
                "annotation1",
            )
            self.assertTrue(reloaded.flush())
            self.assertFalse(reloaded.flush())

    def test_corrupt_image_sidecar_recovers_from_valid_backup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = AnnotationManager(temp_dir, start_autosave=False)
            manager.create_project({
                "id": "project1",
                "name": "测试项目",
                "images": [{
                    "filename": "cat.jpg",
                    "annotations": [],
                }],
                "classes": ["cat"],
            })
            manager.save_annotations("project1", 0, [{"id": "first"}])
            manager.save_annotations("project1", 0, [{"id": "second"}])
            sidecar = next(
                (Path(temp_dir) / "project1" / "image_annotations").glob("*.json")
            )
            self.assertTrue(sidecar.with_suffix(".json.bak").is_file())
            sidecar.write_bytes(b"{")

            recovered = AnnotationManager(temp_dir, start_autosave=False)
            self.assertEqual(
                recovered.get_annotations("project1", 0),
                [{"id": "first"}],
            )
            self.assertEqual(
                json.loads(sidecar.read_text())["annotations"],
                [{"id": "first"}],
            )

    def test_legacy_detail_migrates_to_sidecars_on_flush(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            project_dir = data_dir / "legacy"
            project_dir.mkdir()
            (data_dir / "projects.json").write_text(json.dumps({
                "schema_version": 2,
                "projects": [{
                    "id": "legacy",
                    "name": "旧项目",
                    "image_count": 1,
                    "updated_at": "2026-01-01T00:00:00",
                }],
            }))
            (project_dir / "annotations.json").write_text(json.dumps({
                "schema_version": 2,
                "project_id": "legacy",
                "images": [{
                    "filename": "cat.jpg",
                    "annotated": True,
                    "annotations": [{"id": "legacy-annotation"}],
                }],
                "classes": ["cat"],
                "updated_at": "2026-01-01T00:00:00",
            }))

            manager = AnnotationManager(data_dir, start_autosave=False)
            self.assertEqual(
                manager.get_annotations("legacy", 0)[0]["id"],
                "legacy-annotation",
            )
            self.assertTrue(manager.flush())
            manifest = json.loads(
                (project_dir / "annotations.json").read_text()
            )
            registry = json.loads((data_dir / "projects.json").read_text())
            self.assertEqual(manifest["schema_version"], 3)
            self.assertNotIn("annotations", manifest["images"][0])
            self.assertEqual(registry["schema_version"], 3)
            self.assertEqual(
                len(list((project_dir / "image_annotations").glob("*.json"))),
                1,
            )
            reloaded = AnnotationManager(data_dir, start_autosave=False)
            self.assertEqual(
                reloaded.get_annotations("legacy", 0)[0]["id"],
                "legacy-annotation",
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
            self.assertEqual(
                recovered.get_project("project1")["name"],
                "原始名称",
            )
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
        self.original_access_token = app_module.app.config.get("ACCESS_TOKEN")
        self.original_content_limit = app_module.app.config["MAX_CONTENT_LENGTH"]
        app_module.app.config["ACCESS_TOKEN"] = None
        app_module.app.config.update(TESTING=True)
        self.client = app_module.app.test_client()

    def tearDown(self):
        self.manager.shutdown()
        app_module.annotation_manager = self.original_manager
        app_module.app.config["ACCESS_TOKEN"] = self.original_access_token
        app_module.app.config["MAX_CONTENT_LENGTH"] = self.original_content_limit
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
        self.assertNotIn("annotations", detail["images"][0])
        sidecar = next(
            (
                self.root
                / "data"
                / project["id"]
                / "image_annotations"
            ).glob("*.json")
        )
        self.assertEqual(
            json.loads(sidecar.read_text())["annotations"][0]["id"],
            "a1",
        )

        project_list = self.client.get("/api/project/list")
        self.assertEqual(project_list.status_code, 200)
        summary = project_list.get_json()["projects"][0]
        self.assertNotIn("images", summary)
        self.assertEqual(summary["image_count"], 1)
        self.assertEqual(summary["annotated_count"], 1)

        project_detail = self.client.get(f"/api/project/{project['id']}")
        manifest_image = project_detail.get_json()["project"]["images"][0]
        self.assertNotIn("annotations", manifest_image)
        annotations = self.client.get(
            "/api/annotation/get",
            query_string={
                "project_id": project["id"],
                "image_index": 0,
            },
        )
        self.assertEqual(annotations.get_json()["annotations"][0]["id"], "a1")

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

    def test_annotation_payload_limits_and_geometry_validation(self):
        project = self.create_project()
        self.client.post(
            f"/api/project/{project['id']}/load_images",
            json={"image_dir": str(self.images_dir)},
        )
        invalid_geometry = self.client.post(
            "/api/annotation/save",
            json={
                "project_id": project["id"],
                "image_index": 0,
                "annotations": [{
                    "id": "bad",
                    "bbox": [0, 0, float("inf"), 10],
                }],
            },
        )
        self.assertEqual(invalid_geometry.status_code, 400)
        self.assertIn("有限数字", invalid_geometry.get_json()["error"])

        too_many = self.client.post(
            "/api/annotation/save",
            json={
                "project_id": project["id"],
                "image_index": 0,
                "annotations": [{}] * (
                    app_module.MAX_ANNOTATIONS_PER_IMAGE + 1
                ),
            },
        )
        self.assertEqual(too_many.status_code, 400)
        self.assertIn("标注数量", too_many.get_json()["error"])

    def test_oversized_request_returns_json_413(self):
        app_module.app.config["MAX_CONTENT_LENGTH"] = 128
        response = self.client.post(
            "/api/project/create",
            data=json.dumps({"name": "x" * 256}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 413)
        self.assertTrue(response.is_json)
        self.assertFalse(response.get_json()["success"])

    def test_cross_origin_writes_are_rejected_without_cors(self):
        response = self.client.post(
            "/api/project/create",
            json={"name": "cross-site"},
            headers={"Origin": "https://attacker.example"},
        )
        self.assertEqual(response.status_code, 403)
        self.assertFalse(response.get_json()["success"])

        page = self.client.get(
            "/",
            headers={"Origin": "https://attacker.example"},
        )
        self.assertNotIn("Access-Control-Allow-Origin", page.headers)

    def test_access_token_protects_pages_and_api(self):
        app_module.app.config["ACCESS_TOKEN"] = "lan-secret"
        unauthorized = self.client.get("/api/system/status")
        self.assertEqual(unauthorized.status_code, 401)

        bootstrap = self.client.get(
            "/?access_token=lan-secret",
            follow_redirects=True,
        )
        self.assertEqual(bootstrap.status_code, 200)
        authorized = self.client.get("/api/system/status")
        self.assertEqual(authorized.status_code, 200)

    def test_malformed_json_always_returns_json_error(self):
        for route in (
            "/api/export/preview",
            "/api/export/preview_compare",
            "/api/video/start_session",
            "/api/video/add_prompt",
            "/api/video/propagate",
            "/api/video/close_session",
            "/api/ai/translate",
            "/api/ai/test",
        ):
            with self.subTest(route=route):
                response = self.client.post(
                    route,
                    data="null",
                    content_type="application/json",
                )
                self.assertEqual(response.status_code, 400)
                self.assertTrue(response.is_json)
                self.assertFalse(response.get_json()["success"])

    def test_ai_requests_keep_tls_verification_enabled(self):
        remote = unittest.mock.Mock(status_code=200)
        remote.json.return_value = {
            "choices": [{"message": {"content": "cat"}}],
        }
        with patch.object(app_module.requests, "post", return_value=remote) as post:
            response = self.client.post(
                "/api/ai/translate",
                json={
                    "text": "猫",
                    "api_url": "https://example.com",
                    "api_key": "secret",
                    "model": "model",
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["translated"], "cat")
        self.assertNotIn("verify", post.call_args.kwargs)
        self.assertEqual(
            post.call_args.args[0],
            "https://example.com/v1/chat/completions",
        )

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


class ExporterContractTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source = self.root / "source"
        self.source.mkdir()
        for filename in ("one.png", "two.png"):
            Image.new("RGB", (20, 20), "white").save(self.source / filename)

    def tearDown(self):
        self.temp_dir.cleanup()

    def image_record(self, filename, polygon=None):
        return {
            "filename": filename,
            "path": str(self.source / filename),
            "annotated": True,
            "annotations": [{
                "id": f"annotation-{filename}",
                "class_name": "legacy",
                "polygon": polygon or [[1, 1], [18, 1], [18, 18], [1, 18]],
                "bbox": [1, 1, 18, 18],
            }],
        }

    def test_yolo_reexport_removes_stale_files_and_preserves_old_on_failure(self):
        output = self.root / "yolo"
        exporter = YOLOExporter()
        project = {
            "name": "dataset",
            "classes": ["configured"],
            "images": [
                self.image_record("one.png"),
                self.image_record("two.png"),
            ],
        }
        exporter.export(project, str(output), smooth_level="none")
        result = exporter.export(
            {**project, "images": [self.image_record("one.png")]},
            str(output),
            smooth_level="none",
        )
        self.assertEqual(result["classes"], ["configured", "legacy"])
        self.assertFalse(any(
            "two" in path.name
            for path in output.rglob("*")
            if path.is_file()
        ))
        label = next((output / "labels").rglob("one.txt"))
        self.assertTrue(label.read_text().startswith("1 "))

        invalid = {
            **project,
            "images": [{
                **self.image_record("one.png"),
                "filename": "../escape.png",
            }],
        }
        with self.assertRaisesRegex(ValueError, "文件名无效"):
            exporter.export(invalid, str(output), smooth_level="none")
        self.assertTrue(label.is_file())

    def test_coco_uses_polygon_area_and_actual_export_counts(self):
        output = self.root / "coco"
        project = {
            "name": "dataset",
            "classes": ["configured"],
            "images": [
                self.image_record(
                    "one.png",
                    polygon=[[0, 0], [10, 0], [0, 10]],
                ),
                {
                    **self.image_record("two.png"),
                    "path": str(self.source / "missing.png"),
                },
            ],
        }
        result = COCOExporter().export(
            project,
            str(output),
            smooth_level="none",
        )
        self.assertEqual(sum(result[split] for split in ("train", "val", "test")), 1)
        self.assertEqual(result["total_annotations"], 1)
        annotation_files = list((output / "annotations").glob("instances_*.json"))
        exported = [
            annotation
            for path in annotation_files
            for annotation in json.loads(path.read_text())["annotations"]
        ]
        self.assertEqual(exported[0]["area"], 50.0)
        self.assertEqual(result["classes"], ["configured", "legacy"])


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
