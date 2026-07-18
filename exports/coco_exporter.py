"""
COCO格式导出器
支持标准COCO实例分割格式
"""
import os
import orjson
import shutil
import tempfile
import math
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
from exports.export_utils import (
    bbox_polygon,
    clamp_bbox,
    deterministic_splits,
    oriented_image_size,
)


class COCOExporter:
    """COCO格式导出器"""

    def __init__(self):
        # 平滑参数配置（在mask级别进行形态学平滑）
        self.smooth_params = {
            'none': {'kernel_size': 0, 'simplify_epsilon': 0.002},
            'low': {'kernel_size': 3, 'simplify_epsilon': 0.0015},
            'medium': {'kernel_size': 5, 'simplify_epsilon': 0.001},
            'high': {'kernel_size': 7, 'simplify_epsilon': 0.0008},
            'ultra': {'kernel_size': 11, 'simplify_epsilon': 0.0005},
        }
        self.export_type = 'segment'  # 默认导出类型

    def _smooth_polygon_via_mask(self, polygon: list, kernel_size: int) -> np.ndarray:
        """通过渲染到mask再提取的方式平滑多边形"""
        if kernel_size == 0:
            return np.array(polygon, dtype=np.float64)

        points = np.array(polygon, dtype=np.float64)
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)

        margin = kernel_size * 2 + 10
        x_min = max(0, int(x_min) - margin)
        y_min = max(0, int(y_min) - margin)
        x_max = int(x_max) + margin
        y_max = int(y_max) + margin

        width = x_max - x_min
        height = y_max - y_min

        mask = np.zeros((height, width), dtype=np.uint8)
        shifted_points = points - np.array([x_min, y_min])
        cv2.fillPoly(mask, [shifted_points.astype(np.int32)], 255)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)

        if kernel_size >= 5:
            blur_size = kernel_size | 1
            smoothed = cv2.GaussianBlur(smoothed, (blur_size, blur_size), 0)
            _, smoothed = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        if not contours:
            return points

        largest = max(contours, key=cv2.contourArea)
        new_points = largest.reshape(-1, 2).astype(np.float64)
        new_points += np.array([x_min, y_min])

        return new_points

    def _adaptive_simplify(self, points: np.ndarray, epsilon_factor: float) -> np.ndarray:
        """自适应简化多边形"""
        if len(points) < 3:
            return points

        contour = points.reshape(-1, 1, 2).astype(np.float32)
        perimeter = cv2.arcLength(contour, True)
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        return approx.reshape(-1, 2)

    def smooth_polygon(self, polygon: list, smooth_level: str = 'medium') -> list:
        """对多边形进行平滑处理（通过mask级别的形态学操作）"""
        if not polygon or len(polygon) < 3:
            return polygon

        params = self.smooth_params.get(smooth_level, self.smooth_params['medium'])
        smoothed = self._smooth_polygon_via_mask(polygon, params['kernel_size'])
        result = self._adaptive_simplify(smoothed, params['simplify_epsilon'])

        return result.tolist()

    def export(
        self,
        project: dict,
        output_dir: str,
        export_type: str = 'segment',
        split_ratio: tuple = (0.8, 0.1, 0.1),
        smooth_level: str = 'medium',
        annotation_loader=None,
    ) -> dict:
        """以暂存目录生成 COCO 数据集，成功后替换既有导出。"""
        if export_type not in {'detect', 'segment'}:
            raise ValueError("COCO 导出类型必须是 detect 或 segment")
        if smooth_level not in self.smooth_params:
            raise ValueError("平滑级别无效")
        if (
            len(split_ratio) != 3
            or any(ratio < 0 for ratio in split_ratio)
            or sum(split_ratio) <= 0
        ):
            raise ValueError("数据集拆分比例无效")
        self.current_smooth_level = smooth_level
        self.export_type = export_type
        output_path = Path(output_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        classes = self._resolve_classes(project, annotation_loader)
        images = [
            (index, image)
            for index, image in enumerate(project.get('images', []))
            if image.get(
                'annotated',
                bool(image.get('annotations')),
            )
        ]
        filenames = [str(image.get('filename') or '') for _, image in images]
        if len(filenames) != len(set(filenames)):
            raise ValueError("项目中存在重复图片文件名")
        splits, split_seed = deterministic_splits(
            images,
            split_ratio,
            project.get('export_seed')
            or project.get('id')
            or project.get('name', 'dataset'),
        )
        stats = {
            'train': 0,
            'val': 0,
            'test': 0,
            'total_annotations': 0,
            'converted_bbox_annotations': 0,
        }

        with tempfile.TemporaryDirectory(
            prefix='.sam3-coco-',
            dir=output_path.parent,
        ) as temporary_dir:
            staging_path = Path(temporary_dir)
            for split in ('train', 'val', 'test'):
                (staging_path / split).mkdir()
            annotations_path = staging_path / 'annotations'
            annotations_path.mkdir()
            for split_name, split_images in splits.items():
                coco_data = self._create_coco_structure(project, classes)
                image_count, annotation_count, converted_count = self._export_split(
                    split_images,
                    staging_path,
                    split_name,
                    coco_data,
                    classes,
                    export_type,
                    annotation_loader,
                )
                (
                    annotations_path / f'instances_{split_name}.json'
                ).write_bytes(
                    orjson.dumps(coco_data, option=orjson.OPT_INDENT_2)
                )
                stats[split_name] = image_count
                stats['total_annotations'] += annotation_count
                stats['converted_bbox_annotations'] += converted_count
            self._publish(staging_path, output_path)

        stats['classes'] = classes
        stats['output_dir'] = str(output_path)
        stats['split_seed'] = split_seed
        return stats

    def _resolve_classes(self, project: dict, annotation_loader=None) -> list:
        """保留项目类别顺序，并补入仍被标注引用的类别。"""
        classes = []
        seen = set()
        for class_name in project.get('classes', []):
            class_name = str(class_name).strip()
            if class_name and class_name not in seen:
                seen.add(class_name)
                classes.append(class_name)
        for image_index, image in enumerate(project.get('images', [])):
            if not image.get(
                'annotated',
                bool(image.get('annotations')),
            ):
                continue
            for annotation in self._annotations_for(
                image,
                image_index,
                annotation_loader,
            ):
                class_name = str(
                    annotation.get('class_name')
                    or annotation.get('label')
                    or 'object'
                ).strip()
                if class_name and class_name not in seen:
                    seen.add(class_name)
                    classes.append(class_name)
        return classes

    @staticmethod
    def _annotations_for(
        image: dict,
        image_index: int,
        annotation_loader,
    ) -> list:
        if annotation_loader is None:
            return image.get('annotations', [])
        annotations = annotation_loader(image_index)
        if not isinstance(annotations, list):
            raise ValueError("标注加载器必须返回数组")
        return annotations

    def _create_coco_structure(self, project: dict, classes: list) -> dict:
        """创建COCO基础结构"""
        return {
            'info': {
                'description': project.get('name', 'SAM3 Annotation Dataset'),
                'url': '',
                'version': '1.0',
                'year': datetime.now().year,
                'contributor': 'SAM3 Annotation Tool',
                'date_created': datetime.now().isoformat()
            },
            'licenses': [{
                'id': 1,
                'name': 'Unknown',
                'url': ''
            }],
            'categories': [
                {'id': i + 1, 'name': name, 'supercategory': 'object'}
                for i, name in enumerate(classes)
            ],
            'images': [],
            'annotations': []
        }

    @staticmethod
    def _clamp_polygon(polygon: list, image_width: int, image_height: int) -> list:
        """将不可信多边形限制到图片边界，避免异常尺寸的 mask 分配。"""
        if not isinstance(polygon, list):
            return []
        points = []
        for point in polygon:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                return []
            try:
                x, y = float(point[0]), float(point[1])
            except (TypeError, ValueError):
                return []
            if not math.isfinite(x) or not math.isfinite(y):
                return []
            points.append([
                min(max(x, 0.0), float(image_width)),
                min(max(y, 0.0), float(image_height)),
            ])
        return points

    def _export_split(
        self,
        images: list,
        output_path: Path,
        split: str,
        coco_data: dict,
        classes: list,
        export_type: str = 'segment',
        annotation_loader=None,
    ) -> tuple[int, int, int]:
        """导出单个数据集分割，并返回实际图片和标注数量。"""
        class_to_id = {
            class_name: index + 1
            for index, class_name in enumerate(classes)
        }
        annotation_id = 1
        exported_images = 0
        total_annotations = 0
        converted_bbox_annotations = 0

        for image_index, img_info in images:
            src_path = img_info.get('path')
            if not src_path or not os.path.isfile(src_path):
                continue
            filename = str(img_info.get('filename') or '')
            if not filename or Path(filename).name != filename:
                raise ValueError(f"图片文件名无效: {filename!r}")
            image_width, image_height = oriented_image_size(src_path)
            if image_width <= 0 or image_height <= 0:
                continue
            image_id = exported_images + 1
            shutil.copy2(src_path, output_path / split / filename)
            coco_data['images'].append({
                'id': image_id,
                'file_name': filename,
                'width': image_width,
                'height': image_height,
                'license': 1,
            })
            exported_images += 1

            annotations = self._annotations_for(
                img_info,
                image_index,
                annotation_loader,
            )
            for annotation in annotations:
                class_name = str(
                    annotation.get('class_name')
                    or annotation.get('label')
                    or 'object'
                ).strip()
                coco_annotation = {
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': class_to_id[class_name],
                    'iscrowd': 0,
                }
                if export_type == 'segment':
                    polygon = self._clamp_polygon(
                        annotation.get('polygon', []),
                        image_width,
                        image_height,
                    )
                    converted_from_bbox = False
                    if len(polygon) < 3:
                        bbox = clamp_bbox(
                            annotation.get('bbox'),
                            image_width,
                            image_height,
                            filename,
                        )
                        if bbox is None:
                            continue
                        polygon = bbox_polygon(bbox)
                        converted_from_bbox = True
                    smoothed = self.smooth_polygon(
                        polygon,
                        self.current_smooth_level,
                    )
                    if len(smoothed) < 3:
                        continue
                    points = [
                        [
                            min(max(float(point[0]), 0.0), float(image_width)),
                            min(max(float(point[1]), 0.0), float(image_height)),
                        ]
                        for point in smoothed
                    ]
                    segmentation = [
                        coordinate
                        for point in points
                        for coordinate in point
                    ]
                    x_values = [point[0] for point in points]
                    y_values = [point[1] for point in points]
                    x_min, x_max = min(x_values), max(x_values)
                    y_min, y_max = min(y_values), max(y_values)
                    area = abs(sum(
                        point[0] * points[(index + 1) % len(points)][1]
                        - points[(index + 1) % len(points)][0] * point[1]
                        for index, point in enumerate(points)
                    )) / 2
                    if area <= 0:
                        continue
                    coco_annotation['segmentation'] = [segmentation]
                    coco_annotation['bbox'] = [
                        x_min,
                        y_min,
                        x_max - x_min,
                        y_max - y_min,
                    ]
                    coco_annotation['area'] = area
                    if converted_from_bbox:
                        converted_bbox_annotations += 1
                else:
                    bbox = clamp_bbox(
                        annotation.get('bbox'),
                        image_width,
                        image_height,
                        filename,
                    )
                    if bbox is None:
                        continue
                    x1, y1, x2, y2 = bbox
                    coco_annotation['bbox'] = [
                        x1,
                        y1,
                        x2 - x1,
                        y2 - y1,
                    ]
                    coco_annotation['area'] = (x2 - x1) * (y2 - y1)
                    coco_annotation['segmentation'] = []
                coco_data['annotations'].append(coco_annotation)
                annotation_id += 1
                total_annotations += 1
        return exported_images, total_annotations, converted_bbox_annotations

    @staticmethod
    def _publish(staging_path: Path, output_path: Path):
        output_path.mkdir(parents=True, exist_ok=True)
        for name in ('train', 'val', 'test', 'annotations'):
            destination = output_path / name
            if destination.is_dir():
                shutil.rmtree(destination)
            elif destination.exists():
                destination.unlink()
            os.replace(staging_path / name, destination)
