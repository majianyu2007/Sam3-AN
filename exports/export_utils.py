"""导出器共享的确定性数据集拆分。"""

import hashlib
import math
import random
from PIL import Image


EXIF_ORIENTATION_TAG = 274
SWAPPED_ORIENTATIONS = frozenset({5, 6, 7, 8})


def oriented_image_size(path) -> tuple[int, int]:
    """仅读取图片元数据，并按 EXIF 方向返回显示尺寸，不解码完整像素。"""
    with Image.open(path) as source:
        width, height = source.size
        if source.getexif().get(EXIF_ORIENTATION_TAG) in SWAPPED_ORIENTATIONS:
            return height, width
        return width, height


def clamp_bbox(bbox, image_width: int, image_height: int, filename: str):
    """验证并限制 xyxy 边界框；退化框返回 None。"""
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox[:4])
    except (TypeError, ValueError) as error:
        raise ValueError(f"图片 {filename} 包含无效边界框坐标") from error
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError(f"图片 {filename} 包含非有限边界框坐标")
    x1 = min(max(x1, 0.0), float(image_width))
    x2 = min(max(x2, 0.0), float(image_width))
    y1 = min(max(y1, 0.0), float(image_height))
    y2 = min(max(y2, 0.0), float(image_height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def bbox_polygon(bbox) -> list:
    """将已验证的 xyxy 边界框转换为四点多边形。"""
    x1, y1, x2, y2 = bbox
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]



def deterministic_splits(items: list, split_ratio: tuple, seed_source) -> tuple:
    """按稳定 seed 洗牌，并在样本足够时保证正比例 split 非空。"""
    seed_bytes = hashlib.sha256(str(seed_source).encode("utf-8")).digest()
    seed = int.from_bytes(seed_bytes[:4], "big")
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    ratio_total = sum(split_ratio)
    quotas = [total * ratio / ratio_total for ratio in split_ratio]
    counts = [math.floor(quota) for quota in quotas]
    remainder = total - sum(counts)
    order = sorted(
        range(3),
        key=lambda index: (quotas[index] - counts[index], -index),
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1

    positive_splits = [
        index for index, ratio in enumerate(split_ratio)
        if ratio > 0
    ]
    if total >= len(positive_splits):
        for target in positive_splits:
            if counts[target] > 0:
                continue
            donors = [
                index for index in positive_splits
                if counts[index] > 1
            ]
            if not donors:
                break
            donor = max(donors, key=lambda index: (counts[index], -index))
            counts[donor] -= 1
            counts[target] += 1

    train_end = counts[0]
    val_end = train_end + counts[1]
    return (
        {
            "train": shuffled[:train_end],
            "val": shuffled[train_end:val_end],
            "test": shuffled[val_end:],
        },
        seed,
    )
