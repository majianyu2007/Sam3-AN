import os
import shutil
import tempfile
import math
from pathlib import Path
from PIL import Image, ImageOps
import yaml
import numpy as np
import cv2


class YOLOExporter:
    """YOLO格式导出器"""

    def __init__(self):
        # 平滑参数配置（在mask级别进行形态学平滑）
        self.smooth_params = {
            'none': {'kernel_size': 0, 'simplify_epsilon': 0.002},
            'low': {'kernel_size': 3, 'simplify_epsilon': 0.0015},
            'medium': {'kernel_size': 5, 'simplify_epsilon': 0.001},
            'high': {'kernel_size': 7, 'simplify_epsilon': 0.0008},
            'ultra': {'kernel_size': 11, 'simplify_epsilon': 0.0005},
        }
        self.format_type = 'segment'  # 默认导出类型

    def _smooth_polygon_via_mask(self, polygon: list, kernel_size: int) -> np.ndarray:
        """通过渲染到mask再提取的方式平滑多边形（最有效的方法）

        原理：
        1. 将多边形渲染到临时mask
        2. 对mask进行形态学平滑
        3. 从平滑后的mask提取新轮廓
        """
        if kernel_size == 0:
            return np.array(polygon, dtype=np.float64)

        points = np.array(polygon, dtype=np.float64)

        # 计算边界框，创建合适大小的mask
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)

        # 添加边距
        margin = kernel_size * 2 + 10
        x_min = max(0, int(x_min) - margin)
        y_min = max(0, int(y_min) - margin)
        x_max = int(x_max) + margin
        y_max = int(y_max) + margin

        width = x_max - x_min
        height = y_max - y_min

        # 创建mask并绘制多边形
        mask = np.zeros((height, width), dtype=np.uint8)
        shifted_points = points - np.array([x_min, y_min])
        cv2.fillPoly(mask, [shifted_points.astype(np.int32)], 255)

        # 形态学平滑
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)

        # 高斯模糊 + 阈值
        if kernel_size >= 5:
            blur_size = kernel_size | 1
            smoothed = cv2.GaussianBlur(smoothed, (blur_size, blur_size), 0)
            _, smoothed = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)

        # 提取新轮廓
        contours, _ = cv2.findContours(smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        if not contours:
            return points

        largest = max(contours, key=cv2.contourArea)
        new_points = largest.reshape(-1, 2).astype(np.float64)

        # 还原坐标偏移
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
        """对多边形进行平滑处理（通过mask级别的形态学操作）

        Args:
            polygon: 多边形点列表 [[x, y], ...]
            smooth_level: 平滑级别 'none', 'low', 'medium', 'high', 'ultra'

        Returns:
            平滑后的多边形点列表
        """
        if not polygon or len(polygon) < 3:
            return polygon

        params = self.smooth_params.get(smooth_level, self.smooth_params['medium'])

        # 通过mask渲染的方式平滑（最有效）
        smoothed = self._smooth_polygon_via_mask(polygon, params['kernel_size'])

        # 简化多边形
        result = self._adaptive_simplify(smoothed, params['simplify_epsilon'])

        return result.tolist()

    def export(self, project: dict, output_dir: str,
               format_type: str = 'segment',
               split_ratio: tuple = (0.8, 0.1, 0.1),
               smooth_level: str = 'medium') -> dict:
        """以暂存目录生成 YOLO 数据集，成功后替换既有导出。"""
        if format_type not in {'detect', 'segment'}:
            raise ValueError("YOLO 导出类型必须是 detect 或 segment")
        if smooth_level not in self.smooth_params:
            raise ValueError("平滑级别无效")
        if (
            len(split_ratio) != 3
            or any(ratio < 0 for ratio in split_ratio)
            or sum(split_ratio) <= 0
        ):
            raise ValueError("数据集拆分比例无效")
        self.current_smooth_level = smooth_level
        self.format_type = format_type
        output_path = Path(output_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        classes = self._resolve_classes(project)
        class_to_id = {class_name: index for index, class_name in enumerate(classes)}
        images = [
            image
            for image in project.get('images', [])
            if image.get('annotations')
        ]
        stems = [Path(str(image.get('filename', ''))).stem for image in images]
        if len(stems) != len(set(stems)):
            raise ValueError("存在同名但扩展名不同的图片，YOLO 标签文件会冲突")
        train_end = int(len(images) * split_ratio[0])
        val_end = train_end + int(len(images) * split_ratio[1])
        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:],
        }
        stats = {'train': 0, 'val': 0, 'test': 0, 'total_annotations': 0}

        with tempfile.TemporaryDirectory(
            prefix='.sam3-yolo-',
            dir=output_path.parent,
        ) as temporary_dir:
            staging_path = Path(temporary_dir)
            for split in ('train', 'val', 'test'):
                (staging_path / 'images' / split).mkdir(parents=True)
                (staging_path / 'labels' / split).mkdir(parents=True)
            for split_name, split_images in splits.items():
                for image in split_images:
                    annotation_count = self._export_image(
                        image,
                        staging_path,
                        split_name,
                        class_to_id,
                    )
                    if annotation_count:
                        stats[split_name] += 1
                        stats['total_annotations'] += annotation_count
            self._generate_yaml(
                staging_path,
                classes,
                project.get('name', 'dataset'),
                output_path,
            )
            self._publish(staging_path, output_path)

        stats['classes'] = classes
        stats['output_dir'] = str(output_path)
        return stats

    def _resolve_classes(self, project: dict) -> list:
        """保留项目类别顺序，并补入仍被标注引用的类别。"""
        classes = []
        seen = set()
        for class_name in project.get('classes', []):
            class_name = str(class_name).strip()
            if class_name and class_name not in seen:
                seen.add(class_name)
                classes.append(class_name)
        for image in project.get('images', []):
            for annotation in image.get('annotations', []):
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
    def clamp_polygon(polygon: list, image_width: int, image_height: int) -> list:
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

    def _export_image(self, img_info: dict, output_path: Path,
                      split: str, class_to_id: dict) -> int:
        """导出一张至少含一个有效标签的图片。"""
        src_path = img_info.get('path')
        if not src_path or not os.path.isfile(src_path):
            return 0
        filename = str(img_info.get('filename') or '')
        if not filename or Path(filename).name != filename:
            raise ValueError(f"图片文件名无效: {filename!r}")
        with Image.open(src_path) as source:
            image = ImageOps.exif_transpose(source)
            image_width, image_height = image.size
        if image_width <= 0 or image_height <= 0:
            return 0

        lines = []
        for annotation in img_info.get('annotations', []):
            class_name = str(
                annotation.get('class_name')
                or annotation.get('label')
                or 'object'
            ).strip()
            class_id = class_to_id[class_name]
            if self.format_type == 'segment' and annotation.get('polygon'):
                polygon = self.clamp_polygon(
                    annotation['polygon'],
                    image_width,
                    image_height,
                )
                if len(polygon) < 3:
                    continue
                smoothed = self.smooth_polygon(
                    polygon,
                    self.current_smooth_level,
                )
                if len(smoothed) < 3:
                    continue
                coordinates = []
                for point in smoothed:
                    x = min(max(float(point[0]) / image_width, 0.0), 1.0)
                    y = min(max(float(point[1]) / image_height, 0.0), 1.0)
                    coordinates.extend([f"{x:.6f}", f"{y:.6f}"])
                lines.append(f"{class_id} " + " ".join(coordinates))
                continue

            bbox = annotation.get('bbox', [])
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = (float(value) for value in bbox[:4])
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                raise ValueError(f"图片 {filename} 包含非有限边界框坐标")
            x1 = min(max(x1, 0.0), float(image_width))
            x2 = min(max(x2, 0.0), float(image_width))
            y1 = min(max(y1, 0.0), float(image_height))
            y2 = min(max(y2, 0.0), float(image_height))
            if x2 <= x1 or y2 <= y1:
                continue
            x_center = (x1 + x2) / 2 / image_width
            y_center = (y1 + y2) / 2 / image_height
            width = (x2 - x1) / image_width
            height = (y2 - y1) / image_height
            lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} "
                f"{width:.6f} {height:.6f}"
            )
        if not lines:
            return 0
        label_path = output_path / 'labels' / split / f"{Path(filename).stem}.txt"
        label_path.write_text('\n'.join(lines), encoding='utf-8')
        shutil.copy2(src_path, output_path / 'images' / split / filename)
        return len(lines)

    def _generate_yaml(
        self,
        output_path: Path,
        classes: list,
        dataset_name: str,
        dataset_root: Path,
    ):
        """生成引用最终目录的 YOLO 配置。"""
        data = {
            'path': str(dataset_root.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': {index: name for index, name in enumerate(classes)},
            'nc': len(classes),
        }
        with (output_path / 'data.yaml').open('w', encoding='utf-8') as file:
            yaml.safe_dump(
                data,
                file,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        (output_path / 'classes.txt').write_text(
            '\n'.join(classes),
            encoding='utf-8',
        )

    @staticmethod
    def _publish(staging_path: Path, output_path: Path):
        output_path.mkdir(parents=True, exist_ok=True)
        for name in ('images', 'labels', 'data.yaml', 'classes.txt'):
            destination = output_path / name
            if destination.is_dir():
                shutil.rmtree(destination)
            elif destination.exists():
                destination.unlink()
            os.replace(staging_path / name, destination)
