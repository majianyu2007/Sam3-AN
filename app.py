import os
import sys
import subprocess
import threading
import webbrowser
from pathlib import Path
import atexit
import re
import shutil

# 添加SAM3到路径 (使用本地 SAM_src 目录)
sam3_src = Path(__file__).parent / "SAM_src"
sys.path.insert(0, str(sam3_src))

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import json
import uuid
import requests
from datetime import datetime

from services.sam3_service import SAM3Service, _select_device
from services.annotation_manager import AnnotationManager
from exports.yolo_exporter import YOLOExporter
from exports.coco_exporter import COCOExporter

app = Flask(__name__)
CORS(app)

# 全局配置
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

# 全局服务实例
sam3_service = None
annotation_manager = AnnotationManager()
sam3_service_lock = threading.Lock()


WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def normalize_directory(
    raw_path,
    field_name,
    *,
    required=False,
    must_exist=False,
    create=False,
):
    """展开用户目录并拒绝当前系统无法使用的跨平台路径。"""
    value = str(raw_path or "").strip()
    if not value:
        if required:
            raise ValueError(f"{field_name}不能为空")
        return ""
    if sys.platform != "win32" and (
        WINDOWS_ABSOLUTE_PATH.match(value) or value.startswith("\\\\")
    ):
        raise ValueError(
            f"{field_name}是 Windows 路径，当前 macOS/Linux 无法访问；"
            "请重新选择本机目录"
        )
    path = Path(os.path.expandvars(value)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if must_exist and not path.is_dir():
        raise ValueError(f"{field_name}不存在或不是目录: {path}")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return str(path)


def json_body():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("请求体必须是 JSON 对象")
    return data


def normalize_classes(raw_classes):
    if isinstance(raw_classes, str):
        raw_classes = raw_classes.split(',')
    if not isinstance(raw_classes, list):
        raise ValueError("类别必须是数组或逗号分隔的字符串")
    classes = []
    for item in raw_classes:
        name = str(item).strip()
        if name and name not in classes:
            classes.append(name)
    return classes


def resolve_allowed_image(raw_path):
    value = str(raw_path or "").strip()
    if not value:
        raise ValueError("缺少图片路径")
    if sys.platform != "win32" and WINDOWS_ABSOLUTE_PATH.match(value):
        raise ValueError("当前系统无法访问该 Windows 图片路径")
    image_path = Path(value).expanduser().resolve()
    allowed_dirs = {app.config['UPLOAD_FOLDER'].resolve()}
    for project in annotation_manager.list_projects():
        image_dir = project.get('image_dir')
        if image_dir:
            allowed_dirs.add(Path(image_dir).expanduser().resolve())
    is_allowed = any(
        image_path == directory or directory in image_path.parents
        for directory in allowed_dirs
    )
    if not image_path.is_file() or not is_allowed:
        raise FileNotFoundError("图片不存在或不属于当前项目")
    return image_path


def error_response(message, status=400, **details):
    return jsonify({"success": False, "error": str(message), **details}), status


def get_sam3_service():
    """线程安全地延迟加载 SAM3 服务。"""
    global sam3_service
    if sam3_service is None:
        with sam3_service_lock:
            if sam3_service is None:
                sam3_service = SAM3Service()
    return sam3_service


def shutdown_services():
    """同步保存状态并释放模型资源。"""
    annotation_manager.shutdown()
    if sam3_service is not None:
        sam3_service.shutdown()


atexit.register(shutdown_services)


# ==================== 页面路由 ====================

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')


@app.route('/video')
def video_page():
    """视频标注页面"""
    return render_template('video.html')

# ==================== 系统 API ====================

@app.route('/api/system/status')
def system_status():
    """返回前端可展示的本机运行状态。"""
    checkpoint = Path.cwd() / 'sam3.pt'
    device = _select_device()
    return jsonify({
        'success': True,
        'platform': sys.platform,
        'device': device,
        'device_label': {
            'cuda': 'NVIDIA CUDA',
            'mps': 'Apple Silicon MPS',
            'cpu': 'CPU'
        }[device],
        'checkpoint_ready': checkpoint.is_file(),
        'checkpoint_path': str(checkpoint),
        'checkpoint_size': checkpoint.stat().st_size if checkpoint.is_file() else 0,
        'python_version': sys.version.split()[0]
    })


def choose_native_directory(prompt):
    """在运行 Flask 的本机打开原生目录选择器。"""
    if sys.platform == 'darwin':
        script = '''
on run argv
    try
        return POSIX path of (choose folder with prompt (item 1 of argv))
    on error number -128
        return ""
    end try
end run
'''
        command = ['osascript', '-e', script, prompt]
    elif sys.platform == 'win32':
        script = (
            'Add-Type -AssemblyName System.Windows.Forms; '
            '$dialog = New-Object System.Windows.Forms.FolderBrowserDialog; '
            f'$dialog.Description = {json.dumps(prompt)}; '
            'if ($dialog.ShowDialog() -eq \"OK\") { $dialog.SelectedPath }'
        )
        command = ['powershell', '-NoProfile', '-Command', script]
    elif shutil.which('zenity'):
        command = ['zenity', '--file-selection', '--directory', f'--title={prompt}']
    else:
        raise RuntimeError('当前桌面环境没有可用的原生目录选择器')

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    selected = completed.stdout.strip()
    if not selected:
        return None
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or '目录选择器启动失败')
    return normalize_directory(
        selected,
        '所选目录',
        required=True,
        must_exist=True,
    )


@app.route('/api/system/select-directory', methods=['POST'])
def select_native_directory():
    """打开原生目录选择器；取消不是错误。"""
    try:
        data = json_body()
        purpose = data.get('purpose', 'image')
        prompt = '选择图片目录' if purpose == 'image' else '选择标注输出目录'
        selected = choose_native_directory(prompt)
        return jsonify({
            'success': True,
            'canceled': selected is None,
            'path': selected
        })
    except subprocess.TimeoutExpired:
        return error_response('目录选择超时', 504)
    except Exception as error:
        return error_response(error, 500)


# ==================== 项目管理API ====================

@app.route('/api/project/create', methods=['POST'])
def create_project():
    """创建新项目并规范化本机路径。"""
    try:
        data = json_body()
        project_id = str(uuid.uuid4())[:8]
        name = str(data.get('name') or f'项目_{project_id}').strip()
        image_dir = normalize_directory(
            data.get('image_dir'),
            '图片目录',
            must_exist=bool(data.get('image_dir')),
        )
        output_dir = normalize_directory(
            data.get('output_dir'),
            '输出目录',
            create=bool(data.get('output_dir')),
        )
        project = annotation_manager.create_project({
            'id': project_id,
            'name': name,
            'image_dir': image_dir,
            'output_dir': output_dir,
            'export_format': data.get('export_format', 'yolo'),
            'classes': normalize_classes(data.get('classes', [])),
            'created_at': datetime.now().isoformat(),
            'images': [],
            'current_index': 0
        })
        return jsonify({'success': True, 'project': project}), 201
    except (ValueError, OSError) as error:
        return error_response(error)


@app.route('/api/project/<project_id>', methods=['GET'])
def get_project(project_id):
    """获取项目信息。"""
    project = annotation_manager.get_project(project_id)
    if not project:
        return error_response('项目不存在', 404)
    return jsonify({'success': True, 'project': project})


@app.route('/api/project/<project_id>/update', methods=['POST'])
def update_project(project_id):
    """更新项目信息并验证路径。"""
    if not annotation_manager.get_project(project_id):
        return error_response('项目不存在', 404)
    try:
        data = json_body()
        updates = {}
        if 'name' in data:
            name = str(data['name']).strip()
            if not name:
                raise ValueError('项目名称不能为空')
            updates['name'] = name
        if 'image_dir' in data:
            updates['image_dir'] = normalize_directory(
                data['image_dir'],
                '图片目录',
                must_exist=bool(data['image_dir']),
            )
        if 'output_dir' in data:
            updates['output_dir'] = normalize_directory(
                data['output_dir'],
                '输出目录',
                create=bool(data['output_dir']),
            )
        if 'classes' in data:
            updates['classes'] = normalize_classes(data['classes'])
        updated_project = annotation_manager.update_project(project_id, updates)
        return jsonify({'success': True, 'project': updated_project})
    except (ValueError, OSError) as error:
        return error_response(error)


@app.route('/api/project/<project_id>/delete', methods=['POST'])
def delete_project(project_id):
    """删除项目。"""
    try:
        annotation_manager.delete_project(project_id)
        return jsonify({'success': True, 'message': '项目已删除'})
    except ValueError as error:
        return error_response(error, 404)
    except OSError as error:
        return error_response(error, 500)


@app.route('/api/project/<project_id>/load_images', methods=['POST'])
def load_project_images(project_id):
    """扫描本机图片目录并保留同名图片的已有标注。"""
    if not annotation_manager.get_project(project_id):
        return error_response('项目不存在', 404)
    try:
        data = json_body()
        image_dir = normalize_directory(
            data.get('image_dir'),
            '图片目录',
            required=True,
            must_exist=True,
        )
        directory = Path(image_dir)
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
        files = sorted(
            (
                path for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in extensions
            ),
            key=lambda path: path.name.casefold(),
        )
        images = [
            {
                'filename': path.name,
                'path': str(path.resolve()),
                'annotated': False,
                'annotations': []
            }
            for path in files
        ]
        annotation_manager.update_project_images(project_id, images, image_dir)
        project = annotation_manager.get_project(project_id)
        return jsonify({
            'success': True,
            'count': len(project['images']),
            'images': project['images'],
            'image_dir': image_dir
        })
    except (ValueError, OSError) as error:
        return error_response(error)


@app.route('/api/project/list', methods=['GET'])
def list_projects():
    """列出所有项目"""
    projects = annotation_manager.list_project_summaries()
    return jsonify({'success': True, 'projects': projects})


# ==================== 图片服务API ====================

@app.route('/api/image/serve')
def serve_image():
    """仅提供已登记项目目录内的图片文件。"""
    try:
        return send_file(resolve_allowed_image(request.args.get('path')))
    except ValueError as error:
        return error_response(error)
    except FileNotFoundError as error:
        return error_response(error, 404)


# ==================== SAM3分割API ====================

@app.route('/api/segment/text', methods=['POST'])
def segment_by_text():
    """文本提示分割。"""
    try:
        data = json_body()
        image_path = resolve_allowed_image(data.get('image_path'))
        prompt = str(data.get('prompt', '')).strip()
        if not prompt:
            raise ValueError('文本提示不能为空')
        confidence = float(data.get('confidence', 0.5))
        if not 0 <= confidence <= 1:
            raise ValueError('置信度必须在 0 到 1 之间')
        results = get_sam3_service().segment_by_text(
            str(image_path),
            prompt,
            confidence,
        )
        return jsonify({'success': True, 'results': results})
    except (ValueError, FileNotFoundError) as error:
        return error_response(error)
    except Exception as error:
        return error_response(error, 500)


@app.route('/api/segment/point', methods=['POST'])
def segment_by_point():
    """点击分割。"""
    try:
        data = json_body()
        image_path = resolve_allowed_image(data.get('image_path'))
        points = data.get('points')
        if not isinstance(points, list) or not points:
            raise ValueError('至少需要一个提示点')
        if any(not isinstance(point, list) or len(point) != 3 for point in points):
            raise ValueError('提示点格式必须为 [x, y, label]')
        results = get_sam3_service().segment_by_points(str(image_path), points)
        return jsonify({'success': True, 'results': results})
    except (ValueError, FileNotFoundError) as error:
        return error_response(error)
    except Exception as error:
        return error_response(error, 500)


@app.route('/api/segment/box', methods=['POST'])
def segment_by_box():
    """框选分割。"""
    try:
        data = json_body()
        image_path = resolve_allowed_image(data.get('image_path'))
        boxes = data.get('boxes')
        if not isinstance(boxes, list) or not boxes:
            raise ValueError('至少需要一个提示框')
        if any(
            not isinstance(box, list) or len(box) not in (4, 5)
            for box in boxes
        ):
            raise ValueError('提示框格式必须为 [x1, y1, x2, y2, label?]')
        results = get_sam3_service().segment_by_boxes(str(image_path), boxes)
        return jsonify({'success': True, 'results': results})
    except (ValueError, FileNotFoundError) as error:
        return error_response(error)
    except Exception as error:
        return error_response(error, 500)


@app.route('/api/segment/batch', methods=['POST'])
def batch_segment():
    """服务端批量分割，逐张保存并返回失败明细。"""
    try:
        data = json_body()
        project_id = data.get('project_id')
        project = annotation_manager.get_project(project_id)
        if not project:
            return error_response('项目不存在', 404)
        prompt = str(data.get('prompt', '')).strip()
        if not prompt:
            raise ValueError('文本提示不能为空')
        class_name = str(data.get('class_name') or prompt).strip()
        start_index = int(data.get('start_index', 0))
        end_index = int(data.get('end_index', -1))
        confidence = float(data.get('confidence', 0.5))
        if start_index < 0 or end_index < -1 or not 0 <= confidence <= 1:
            raise ValueError('批量分割参数无效')

        images = project.get('images', [])
        if end_index == -1:
            end_index = len(images)
        end_index = min(end_index, len(images))
        service = get_sam3_service()
        processed = 0
        total_detections = 0
        results = []
        errors = []

        for index in range(start_index, end_index):
            image = images[index]
            if data.get('skip_annotated', True) and image.get('annotated', False):
                continue
            try:
                detections = service.segment_by_text(
                    image['path'],
                    prompt,
                    confidence,
                )
                for detection in detections:
                    detection['class_name'] = class_name
                if detections:
                    annotation_manager.add_annotations(
                        project_id,
                        index,
                        detections,
                        class_name,
                    )
                processed += 1
                total_detections += len(detections)
                results.append({
                    'index': index,
                    'filename': image['filename'],
                    'count': len(detections)
                })
            except Exception as error:
                print(f"[ERROR] 批量分割图片 {image['filename']} 失败: {error}")
                errors.append({
                    'index': index,
                    'filename': image['filename'],
                    'error': str(error)
                })

        return jsonify({
            'success': True,
            'processed': processed,
            'failed': len(errors),
            'total_detections': total_detections,
            'results': results,
            'errors': errors
        })
    except ValueError as error:
        return error_response(error)
    except Exception as error:
        return error_response(error, 500)


# ==================== 标注管理API ====================

@app.route('/api/annotation/save', methods=['POST'])
def save_annotation():
    """立即持久化当前图片的完整标注列表。"""
    try:
        data = json_body()
        annotations = data.get('annotations', [])
        if not isinstance(annotations, list):
            raise ValueError('annotations 必须是数组')
        annotation_manager.save_annotations(
            data.get('project_id'),
            data.get('image_index'),
            annotations,
        )
        return jsonify({'success': True})
    except ValueError as error:
        return error_response(error)
    except OSError as error:
        return error_response(f'写入标注失败: {error}', 500)


@app.route('/api/annotation/get', methods=['GET'])
def get_annotation():
    """获取标注。"""
    try:
        project_id = request.args.get('project_id')
        image_index = int(request.args.get('image_index', 0))
        annotations = annotation_manager.get_annotations(project_id, image_index)
        return jsonify({'success': True, 'annotations': annotations})
    except (TypeError, ValueError) as error:
        return error_response(error)


@app.route('/api/annotation/update', methods=['POST'])
def update_annotation():
    """更新单个标注。"""
    try:
        data = json_body()
        updates = data.get('updates', {})
        if not isinstance(updates, dict):
            raise ValueError('updates 必须是对象')
        annotation_manager.update_annotation(
            data.get('project_id'),
            data.get('image_index'),
            data.get('annotation_id'),
            updates,
        )
        return jsonify({'success': True})
    except ValueError as error:
        return error_response(error)
    except OSError as error:
        return error_response(f'写入标注失败: {error}', 500)


@app.route('/api/annotation/delete', methods=['POST'])
def delete_annotation():
    """删除标注。"""
    try:
        data = json_body()
        annotation_manager.delete_annotation(
            data.get('project_id'),
            data.get('image_index'),
            data.get('annotation_id'),
        )
        return jsonify({'success': True})
    except ValueError as error:
        return error_response(error)
    except OSError as error:
        return error_response(f'写入标注失败: {error}', 500)


# ==================== 类别管理API ====================

@app.route('/api/classes/update', methods=['POST'])
def update_classes():
    """更新去重后的类别列表。"""
    try:
        data = json_body()
        annotation_manager.update_classes(
            data.get('project_id'),
            normalize_classes(data.get('classes', [])),
        )
        return jsonify({'success': True})
    except ValueError as error:
        return error_response(error)
    except OSError as error:
        return error_response(f'写入类别失败: {error}', 500)


# ==================== 导出API ====================

@app.route('/api/export/yolo', methods=['POST'])
def export_yolo():
    """导出 YOLO 数据集。"""
    try:
        data = json_body()
        project = annotation_manager.get_project(data.get('project_id'))
        if not project:
            return error_response('项目不存在', 404)
        output_dir = normalize_directory(
            data.get('output_dir'),
            '输出目录',
            required=True,
            create=True,
        )
        result = YOLOExporter().export(
            project,
            output_dir,
            format_type=data.get('export_type', 'segment'),
            smooth_level=data.get('smooth_level', 'medium'),
        )
        return jsonify({'success': True, 'result': result})
    except ValueError as error:
        return error_response(error)
    except Exception as error:
        return error_response(error, 500)


@app.route('/api/export/coco', methods=['POST'])
def export_coco():
    """导出 COCO 数据集。"""
    try:
        data = json_body()
        project = annotation_manager.get_project(data.get('project_id'))
        if not project:
            return error_response('项目不存在', 404)
        output_dir = normalize_directory(
            data.get('output_dir'),
            '输出目录',
            required=True,
            create=True,
        )
        result = COCOExporter().export(
            project,
            output_dir,
            export_type=data.get('export_type', 'segment'),
            smooth_level=data.get('smooth_level', 'medium'),
        )
        return jsonify({'success': True, 'result': result})
    except ValueError as error:
        return error_response(error)
    except Exception as error:
        return error_response(error, 500)


# ==================== 导出预览API ====================

@app.route('/api/export/preview', methods=['POST'])
def export_preview():
    """生成导出预览图片，显示平滑后的分割覆盖效果"""
    import cv2
    import numpy as np
    import base64
    from io import BytesIO

    data = request.json
    project_id = data.get('project_id')
    image_index = data.get('image_index', 0)
    smooth_level = data.get('smooth_level', 'medium')
    show_polygon = data.get('show_polygon', True)
    show_fill = data.get('show_fill', True)
    opacity = data.get('opacity', 0.4)

    try:
        project = annotation_manager.get_project(project_id)
        if not project:
            return jsonify({'success': False, 'error': '项目不存在'})

        images = project.get('images', [])
        if image_index >= len(images):
            return jsonify({'success': False, 'error': '图片索引超出范围'})

        img_info = images[image_index]
        image_path = img_info.get('path')

        if not image_path or not os.path.exists(image_path):
            return jsonify({'success': False, 'error': '图片文件不存在'})

        # 读取原始图片
        img = cv2.imread(image_path)
        if img is None:
            return jsonify({'success': False, 'error': '无法读取图片'})

        overlay = img.copy()
        annotations = img_info.get('annotations', [])

        # 使用导出器的平滑方法
        exporter = YOLOExporter()

        # 颜色列表（BGR格式）
        colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 品红
            (0, 255, 255),  # 黄色
            (128, 0, 255),  # 紫色
            (255, 128, 0),  # 橙色
        ]

        for i, ann in enumerate(annotations):
            polygon = ann.get('polygon', [])
            if not polygon or len(polygon) < 3:
                continue

            # 应用平滑处理
            smoothed_polygon = exporter.smooth_polygon(polygon, smooth_level)

            # 转换为numpy数组
            pts = np.array(smoothed_polygon, dtype=np.int32)
            color = colors[i % len(colors)]

            # 绘制填充
            if show_fill:
                cv2.fillPoly(overlay, [pts], color)

            # 绘制轮廓线
            if show_polygon:
                cv2.polylines(img, [pts], True, color, 2)

        # 混合原图和覆盖层
        if show_fill:
            img = cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0)

        # 添加标注信息文字
        for i, ann in enumerate(annotations):
            polygon = ann.get('polygon', [])
            if not polygon:
                continue

            smoothed_polygon = exporter.smooth_polygon(polygon, smooth_level)
            if smoothed_polygon:
                # 计算中心点
                pts = np.array(smoothed_polygon)
                cx = int(pts[:, 0].mean())
                cy = int(pts[:, 1].mean())

                label = ann.get('class_name') or ann.get('label', '')
                color = colors[i % len(colors)]

                # 绘制标签背景
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (cx - 2, cy - text_h - 4), (cx + text_w + 2, cy + 2), color, -1)
                cv2.putText(img, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 转换为base64
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # 统计信息
        stats = {
            'total_annotations': len(annotations),
            'smooth_level': smooth_level,
            'image_size': [img.shape[1], img.shape[0]],
            'filename': img_info.get('filename', '')
        }

        return jsonify({
            'success': True,
            'preview': f'data:image/jpeg;base64,{img_base64}',
            'stats': stats
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/export/preview_compare', methods=['POST'])
def export_preview_compare():
    """生成多个平滑级别的对比预览"""
    import cv2
    import numpy as np
    import base64

    data = request.json
    project_id = data.get('project_id')
    image_index = data.get('image_index', 0)
    annotation_index = data.get('annotation_index', 0)  # 指定要预览的标注索引

    try:
        project = annotation_manager.get_project(project_id)
        if not project:
            return jsonify({'success': False, 'error': '项目不存在'})

        images = project.get('images', [])
        if image_index >= len(images):
            return jsonify({'success': False, 'error': '图片索引超出范围'})

        img_info = images[image_index]
        image_path = img_info.get('path')
        annotations = img_info.get('annotations', [])

        if annotation_index >= len(annotations):
            return jsonify({'success': False, 'error': '标注索引超出范围'})

        if not image_path or not os.path.exists(image_path):
            return jsonify({'success': False, 'error': '图片文件不存在'})

        # 读取原始图片
        original_img = cv2.imread(image_path)
        if original_img is None:
            return jsonify({'success': False, 'error': '无法读取图片'})

        exporter = YOLOExporter()
        polygon = annotations[annotation_index].get('polygon', [])

        if not polygon or len(polygon) < 3:
            return jsonify({'success': False, 'error': '标注没有有效的多边形数据'})

        # 生成不同平滑级别的预览
        levels = ['none', 'low', 'medium', 'high', 'ultra']
        previews = {}

        for level in levels:
            img = original_img.copy()
            smoothed_polygon = exporter.smooth_polygon(polygon, level)
            pts = np.array(smoothed_polygon, dtype=np.int32)

            # 绘制填充和轮廓
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)

            # 添加级别标签
            cv2.putText(img, f'{level} ({len(smoothed_polygon)} pts)',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 转换为base64
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            previews[level] = f'data:image/jpeg;base64,{base64.b64encode(buffer).decode("utf-8")}'

        return jsonify({
            'success': True,
            'previews': previews,
            'original_points': len(polygon),
            'annotation_label': annotations[annotation_index].get('class_name', '')
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


# ==================== 视频分割API ====================

@app.route('/api/video/start_session', methods=['POST'])
def video_start_session():
    """开始视频分割会话"""
    data = request.json
    video_path = data.get('video_path')

    try:
        service = get_sam3_service()
        session_id = service.start_video_session(video_path)
        return jsonify({'success': True, 'session_id': session_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/video/add_prompt', methods=['POST'])
def video_add_prompt():
    """添加视频分割提示"""
    data = request.json
    session_id = data.get('session_id')
    frame_index = data.get('frame_index', 0)
    prompt_type = data.get('prompt_type', 'text')
    prompt_data = data.get('prompt_data')

    try:
        service = get_sam3_service()
        results = service.add_video_prompt(
            session_id, frame_index, prompt_type, prompt_data
        )
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/video/propagate', methods=['POST'])
def video_propagate():
    """传播视频分割"""
    data = request.json
    session_id = data.get('session_id')

    try:
        service = get_sam3_service()
        results = service.propagate_video(session_id)
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/video/close_session', methods=['POST'])
def video_close_session():
    """关闭视频会话"""
    data = request.json
    session_id = data.get('session_id')

    try:
        service = get_sam3_service()
        service.close_video_session(session_id)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ==================== AI翻译API ====================

@app.route('/api/ai/translate', methods=['POST'])
def ai_translate():
    """
    使用OpenAI格式的API将中文翻译成简短的英文
    支持第三方API（如DeepSeek、通义千问、Moonshot等）
    """
    data = request.json
    text = data.get('text', '').strip()
    api_url = data.get('api_url', '').strip()
    api_key = data.get('api_key', '').strip()
    model = data.get('model', 'gpt-3.5-turbo').strip()

    if not text:
        return jsonify({'success': False, 'error': '文本为空'})

    if not api_url or not api_key:
        return jsonify({'success': False, 'error': 'API未配置'})

    # 确保API URL以/v1/chat/completions结尾
    if not api_url.endswith('/v1/chat/completions'):
        api_url = api_url.rstrip('/')
        if not api_url.endswith('/v1'):
            api_url += '/v1'
        api_url += '/chat/completions'

    try:
        # 构建翻译提示
        system_prompt = """You are a translation assistant for image segmentation tasks.
Translate the user's Chinese text into simple, concise English words or short phrases that can be used as object detection prompts.
Rules:
1. Output ONLY the English translation, nothing else
2. Keep it as short as possible (1-3 words preferred)
3. Use common object names (e.g., "apple", "car", "person", "red ball")
4. If multiple objects, separate with comma
5. No explanations, no quotes, just the words"""

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }

        payload = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': text}
            ],
            'max_tokens': 100,
            'temperature': 0.3
        }

        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        print(f"[AI翻译] 正在连接: {api_url}")

        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=30,
            verify=False  # 跳过SSL证书验证，解决WSL环境下的证书问题
        )

        print(f"[AI翻译] 响应状态码: {response.status_code}")

        if response.status_code != 200:
            error_msg = f'API请求失败: {response.status_code}'
            try:
                error_data = response.json()
                if 'error' in error_data:
                    error_msg = error_data['error'].get('message', error_msg)
            except:
                pass
            return jsonify({'success': False, 'error': error_msg})

        result = response.json()
        translated = result['choices'][0]['message']['content'].strip()

        print(f"[AI翻译] {text} -> {translated}")

        return jsonify({
            'success': True,
            'original': text,
            'translated': translated
        })

    except requests.exceptions.Timeout:
        return jsonify({'success': False, 'error': 'API请求超时 (30秒)'})
    except requests.exceptions.SSLError as e:
        print(f"[AI翻译] SSL错误: {e}")
        return jsonify({'success': False, 'error': f'SSL证书错误'})
    except requests.exceptions.ConnectionError as e:
        print(f"[AI翻译] 连接错误: {e}")
        return jsonify({'success': False, 'error': '无法连接到API服务器'})
    except Exception as e:
        print(f"[AI翻译错误] {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/ai/test', methods=['POST'])
def ai_test():
    """测试AI API配置是否有效"""
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    data = request.json
    api_url = data.get('api_url', '').strip()
    api_key = data.get('api_key', '').strip()
    model = data.get('model', 'gpt-3.5-turbo').strip()

    if not api_url or not api_key:
        return jsonify({'success': False, 'error': 'API地址和密钥不能为空'})

    # 确保API URL格式正确
    if not api_url.endswith('/v1/chat/completions'):
        api_url = api_url.rstrip('/')
        if not api_url.endswith('/v1'):
            api_url += '/v1'
        api_url += '/chat/completions'

    try:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }

        payload = {
            'model': model,
            'messages': [
                {'role': 'user', 'content': 'Hello'}
            ],
            'max_tokens': 10
        }

        print(f"[AI测试] 正在连接: {api_url}")

        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=30,
            verify=False  # 跳过SSL证书验证，解决WSL环境下的证书问题
        )

        print(f"[AI测试] 响应状态码: {response.status_code}")

        if response.status_code == 200:
            return jsonify({'success': True, 'message': 'API连接成功'})
        else:
            error_msg = f'状态码: {response.status_code}'
            try:
                error_data = response.json()
                if 'error' in error_data:
                    error_msg = error_data['error'].get('message', error_msg)
            except:
                pass
            return jsonify({'success': False, 'error': error_msg})

    except requests.exceptions.Timeout:
        return jsonify({'success': False, 'error': '连接超时 (30秒)'})
    except requests.exceptions.SSLError as e:
        print(f"[AI测试] SSL错误: {e}")
        return jsonify({'success': False, 'error': f'SSL证书错误: {str(e)[:100]}'})
    except requests.exceptions.ConnectionError as e:
        print(f"[AI测试] 连接错误: {e}")
        return jsonify({'success': False, 'error': f'无法连接到API服务器，请检查网络或API地址是否正确'})
    except Exception as e:
        print(f"[AI测试] 未知错误: {e}")
        return jsonify({'success': False, 'error': str(e)})


def wait_for_server(url, timeout=30):
    """等待服务器启动就绪"""
    import time
    import urllib.request
    import urllib.error

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except (urllib.error.URLError, urllib.error.HTTPError):
            time.sleep(0.3)
    return False


def open_browser(url):
    """等待服务就绪后打开浏览器（独立窗口模式）"""
    print("[INFO] 等待服务启动...")

    # 等待服务就绪
    if not wait_for_server(url):
        print("[ERROR] 服务启动超时，请手动打开浏览器访问:", url)
        return

    print("[INFO] 服务已就绪，正在打开浏览器...")

    # 按平台选择已安装的 Chromium 内核浏览器（用于 --app 独立窗口模式）
    chrome_paths = []
    if sys.platform == "win32":
        chrome_paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ]
    elif sys.platform == "darwin":
        chrome_paths = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ]

    browser_path = None
    for path in chrome_paths:
        if os.path.exists(path):
            browser_path = path
            break

    if browser_path:
        # 使用 --app 模式打开，类似独立应用（无地址栏）
        subprocess.Popen([
            browser_path,
            f'--app={url}',
            '--disable-infobars',
            '--no-first-run',
            '--force-device-scale-factor=1',  # 强制缩放比例为1，避免字体变小
        ])
        print(f"[INFO] 已在独立窗口中打开: {url}")
    else:
        # 其他平台（Linux 等）或未检测到 Chrome/Edge，使用系统默认浏览器
        webbrowser.open(url)
        print(f"[INFO] 已在默认浏览器中打开: {url}")


# 退出程序的API
@app.route('/api/app/exit', methods=['POST'])
def exit_app():
    """先同步落盘，再在响应发出后停止本地服务。"""
    try:
        shutdown_services()
    except Exception as error:
        return error_response(f'退出前保存失败: {error}', 500)

    shutdown = request.environ.get('werkzeug.server.shutdown')
    if shutdown is not None:
        threading.Timer(0.1, shutdown).start()
    else:
        threading.Timer(0.2, os._exit, args=(0,)).start()
    return jsonify({'success': True})


def _find_available_port(preferred=(5000, 5001, 5055, 8000, 8080)):
    """选择可用端口。

    macOS Sonoma/Sequoia 默认占用 5000 端口（AirPlay Receiver），按候选顺序
    回退；全部占用时让操作系统分配空闲端口。
    """
    import socket
    for port in preferred:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("0.0.0.0", port))
                return port
        except OSError:
            continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        return s.getsockname()[1]


if __name__ == '__main__':
    print("=" * 50)
    print("SAM3 AN - 数据标注工具")
    print("=" * 50)

    # 在后台线程中等待服务就绪后打开浏览器
    port = _find_available_port()
    url = f"http://localhost:{port}"
    if port != 5000:
        print(f"[INFO] 默认端口 5000 被占用（macOS AirPlay Receiver 常见），改用端口 {port}")

    print(f"[INFO] 正在启动服务器...")
    print(f"[INFO] 服务就绪后将自动打开浏览器")
    print("=" * 50)

    # 启动Flask服务器（关闭debug模式以避免重复打开浏览器）
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
