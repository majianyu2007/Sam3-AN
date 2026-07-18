/**
 * SAM3 数据标注工具 - 前端交互逻辑
 */

// 全局状态
const state = {
    projectId: null,
    imageDir: '',
    outputDir: '',
    images: [],
    annotatedCount: 0,
    currentIndex: 0,
    annotations: [],
    classes: [],
    currentClass: null,  // 当前选中的类名
    currentTool: 'point',
    isPositive: true,
    confidence: 0.5,
    zoom: 1,
    pan: { x: 0, y: 0 },
    drawing: false,
    drawStart: null,
    panning: false,      // 平移拖动中
    panStart: null,      // 拖动起始位置
    spacePressed: false, // 空格键按下状态
    selectedAnnotation: null,
    tempPoints: [],
    tempBoxes: [],  // 临时框数组，每个框包含 {x1, y1, x2, y2, label}
    tempPolygon: [], // 手动绘制的多边形顶点
    dirty: false,
    revision: 0,
    history: [],
    historyIndex: -1,
    historyWeights: [],
    historyWeight: 0,
};

// Canvas相关
let canvas, ctx;
let currentImage = null;

// 性能优化：缓存和帧控制
let staticCanvas = null;  // 离屏canvas缓存静态内容
let staticCtx = null;
let staticCacheDirty = true;  // 静态缓存是否需要更新
let rafId = null;  // requestAnimationFrame ID
let pendingDraw = null;  // 待绘制的动态内容
const IMAGE_ROW_HEIGHT = 40;
let imageFilterQuery = '';
let filteredImageIndices = [];
let renderedImages = null;
let imageListRafId = null;
let filteredImagesSource = null;
let filteredImagesQuery = null;
const ANNOTATION_ROW_HEIGHT = 62;
let annotationListRafId = null;
let renderedAnnotationImageKey = null;
let imageSearchTimer = null;
let imageLoadToken = 0;
let projectLoadToken = 0;
let imageLoading = false;
let exportPreviewController = null;
let exportPreviewObjectUrl = null;
const MAX_ANNOTATION_HISTORY = 30;
const MAX_ANNOTATION_HISTORY_WEIGHT = 500000;
let annotationAutosaveTimer = null;
let saveQueue = Promise.resolve();
let lastQueuedSave = null;
let batchCancelRequested = false;

function replaceImages(images) {
    state.images = Array.isArray(images) ? images : [];
    state.annotatedCount = state.images.reduce(
        (count, image) => count + (image.annotated ? 1 : 0),
        0
    );
    filteredImagesSource = null;
    renderedImages = null;
}

function setImageAnnotated(index, annotated) {
    const image = state.images[index];
    if (!image) return;
    const nextValue = Boolean(annotated);
    if (Boolean(image.annotated) !== nextValue) {
        state.annotatedCount += nextValue ? 1 : -1;
        image.annotated = nextValue;
    }
}

function cancelBatch() {
    batchCancelRequested = true;
    const button = document.getElementById('cancelBatchButton');
    button.disabled = true;
    button.textContent = '正在完成当前图片…';
}

function updateSaveIndicator(status, message) {
    const indicator = document.getElementById('saveStatus');
    const button = document.getElementById('saveButton');
    if (!indicator || !button) return;
    const labels = {
        saved: '已保存',
        dirty: '等待自动保存',
        saving: '正在保存…',
        error: '保存失败'
    };
    indicator.className = `save-status ${status}`;
    indicator.textContent = message || labels[status] || '';
    button.disabled = status === 'saving';
}

function renderAnnotationState() {
    setImageAnnotated(state.currentIndex, state.annotations.length > 0);
    state.selectedAnnotation = null;
    invalidateStaticCache();
    updateAnnotationList();
    updateImageList();
    redraw();
}

function annotationHistoryWeight(annotations) {
    let weight = annotations.length * 8;
    for (const annotation of annotations) {
        weight += Array.isArray(annotation.polygon)
            ? annotation.polygon.length
            : 0;
    }
    return weight;
}

function resetAnnotationHistory() {
    state.revision++;
    const snapshot = structuredClone(state.annotations);
    const weight = annotationHistoryWeight(snapshot);
    state.history = [snapshot];
    state.historyWeights = [weight];
    state.historyWeight = weight;
    state.historyIndex = 0;
    state.dirty = false;
    updateSaveIndicator('saved');
    updateHistoryButtons();
}

function recordAnnotationMutation({ autosave = true } = {}) {
    state.revision++;
    const discardedWeights = state.historyWeights.splice(state.historyIndex + 1);
    state.historyWeight -= discardedWeights.reduce(
        (total, weight) => total + weight,
        0
    );
    state.history.splice(state.historyIndex + 1);
    const snapshot = structuredClone(state.annotations);
    const weight = annotationHistoryWeight(snapshot);
    state.history.push(snapshot);
    state.historyWeights.push(weight);
    state.historyWeight += weight;
    while (
        state.history.length > 2
        && (
            state.history.length > MAX_ANNOTATION_HISTORY
            || state.historyWeight > MAX_ANNOTATION_HISTORY_WEIGHT
        )
    ) {
        state.history.shift();
        state.historyWeight -= state.historyWeights.shift();
    }
    state.historyIndex = state.history.length - 1;
    state.dirty = true;
    renderAnnotationState();
    updateSaveIndicator('dirty');
    updateHistoryButtons();
    if (autosave) scheduleAnnotationAutosave();
}

function updateHistoryButtons() {
    const undoButton = document.getElementById('undoButton');
    const redoButton = document.getElementById('redoButton');
    if (undoButton) undoButton.disabled = state.historyIndex <= 0;
    if (redoButton) redoButton.disabled =
        state.historyIndex < 0 || state.historyIndex >= state.history.length - 1;
}

function applyHistory(index) {
    if (index < 0 || index >= state.history.length) return;
    state.historyIndex = index;
    state.annotations = structuredClone(state.history[index]);
    state.revision++;
    state.dirty = true;
    renderAnnotationState();
    updateSaveIndicator('dirty');
    updateHistoryButtons();
    scheduleAnnotationAutosave();
}

function undoAnnotationChange() {
    if (state.tempPolygon.length > 0) {
        undoPolygonPoint();
        return;
    }
    applyHistory(state.historyIndex - 1);
}

function redoAnnotationChange() {
    applyHistory(state.historyIndex + 1);
}

// 颜色映射
const colors = [
    '#e94560', '#4dabf7', '#4ade80', '#fbbf24', '#a78bfa',
    '#f472b6', '#22d3d8', '#fb923c', '#84cc16', '#6366f1'
];

async function apiRequest(url, options = {}) {
    const response = await fetch(url, options);
    let data;
    try {
        data = await response.json();
    } catch {
        throw new Error(`服务返回了无法解析的响应（HTTP ${response.status}）`);
    }
    if (!response.ok || data.success === false) {
        throw new Error(data.error || `请求失败（HTTP ${response.status}）`);
    }
    return data;
}

async function imageRequest(url, options = {}) {
    const response = await fetch(url, options);
    if (!response.ok) {
        let message = `请求失败（HTTP ${response.status}）`;
        try {
            const data = await response.json();
            message = data.error || message;
        } catch {
            // 二进制接口的错误仍应由统一 JSON 信封返回。
        }
        throw new Error(message);
    }
    const contentType = response.headers.get('Content-Type') || '';
    if (!contentType.startsWith('image/')) {
        throw new Error('服务未返回预览图片');
    }
    let stats = {};
    const encodedStats = response.headers.get('X-SAM3-Preview-Stats');
    if (encodedStats) {
        try {
            stats = JSON.parse(decodeURIComponent(encodedStats));
        } catch {
            throw new Error('预览统计信息格式无效');
        }
    }
    return { blob: await response.blob(), stats };
}

function releaseExportPreviewObjectUrl() {
    if (exportPreviewObjectUrl) {
        URL.revokeObjectURL(exportPreviewObjectUrl);
        exportPreviewObjectUrl = null;
    }
}

function applyAccessibleButtonNames(root = document) {
    root.querySelectorAll('button[title]:not([aria-label])').forEach(button => {
        if (!button.textContent.trim()) {
            button.setAttribute('aria-label', button.title);
        }
    });
}
function cancelPreviewRequests() {
    exportPreviewController?.abort();
    exportPreviewController = null;
    releaseExportPreviewObjectUrl();
    const previewImage = document.getElementById('exportPreviewImage');
    if (previewImage) {
        previewImage.removeAttribute('src');
        previewImage.style.display = 'none';
    }
}



let confirmationResolver = null;

function resolveConfirmation(value) {
    const resolver = confirmationResolver;
    confirmationResolver = null;
    if (resolver) resolver(value);

    const modalElement = document.getElementById('confirmModal');
    const modal = bootstrap.Modal.getOrCreateInstance(modalElement);
    if (modal._isTransitioning) {
        modalElement.addEventListener('shown.bs.modal', () => modal.hide(), {
            once: true
        });
    } else {
        modal.hide();
    }
}

function confirmAction({
    title = '确认操作',
    message,
    confirmText = '确认',
    danger = false
}) {
    const modalElement = document.getElementById('confirmModal');
    if (!modalElement.dataset.initialized) {
        modalElement.dataset.initialized = 'true';
        modalElement.addEventListener('hidden.bs.modal', () => {
            if (confirmationResolver) {
                const resolver = confirmationResolver;
                confirmationResolver = null;
                resolver(false);
            }
        });
    }
    if (confirmationResolver) {
        confirmationResolver(false);
        confirmationResolver = null;
    }
    document.getElementById('confirmModalTitle').textContent = title;
    document.getElementById('confirmModalMessage').textContent = message;
    const confirmButton = document.getElementById('confirmModalButton');
    confirmButton.textContent = confirmText;
    confirmButton.className = `btn ${danger ? 'btn-danger' : 'btn-primary'}`;
    bootstrap.Modal.getOrCreateInstance(modalElement).show();
    return new Promise(resolve => {
        confirmationResolver = resolve;
    });
}

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, char => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    })[char]);
}

function formatBytes(bytes) {
    if (!bytes) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    return `${(bytes / (1024 ** index)).toFixed(index > 1 ? 1 : 0)} ${units[index]}`;
}

async function loadRuntimeStatus() {
    const element = document.getElementById('runtimeStatus');
    try {
        const status = await apiRequest('/api/system/status');
        const ready = status.checkpoint_ready;
        element.className = `runtime-status ${ready ? 'ready' : 'warning'}`;
        element.querySelector('span:last-child').textContent =
            `${status.device_label} · ${ready ? formatBytes(status.checkpoint_size) : '缺少 sam3.pt'}`;
        element.title = [
            `Python ${status.python_version}`,
            `设备: ${status.device_label}`,
            `权重: ${status.checkpoint_path}`
        ].join('\n');
    } catch (error) {
        element.className = 'runtime-status error';
        element.querySelector('span:last-child').textContent = '运行环境检测失败';
        element.title = error.message;
    }
}

async function chooseDirectory(inputId, purpose) {
    const input = document.getElementById(inputId);
    if (!input) return;
    try {
        const data = await apiRequest('/api/system/select-directory', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ purpose })
        });
        if (!data.canceled && data.path) {
            input.value = data.path;
            input.dispatchEvent(new Event('change', { bubbles: true }));
        }
    } catch (error) {
        showToast('目录选择失败', error.message, 'danger');
    }
}

// 初始化：先绑定 UI，再按顺序恢复服务端与本地状态。
document.addEventListener('DOMContentLoaded', async () => {
    initCanvas();
    initEventListeners();
    restorePanelState();
    handleResponsiveCollapse();
    await Promise.all([loadRuntimeStatus(), loadProjects()]);
    await restoreWorkState();
});

window.addEventListener('beforeunload', event => {
    if (!state.dirty || !state.projectId || !state.images[state.currentIndex]) return;
    fetch('/api/annotation/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            project_id: state.projectId,
            image_index: state.currentIndex,
            annotations: state.annotations
        }),
        keepalive: true
    }).catch(() => {});
    event.preventDefault();
    event.returnValue = '';
});

document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden' && state.dirty) {
        saveAnnotations(false);
    }
});

// 保存工作状态到localStorage
function saveWorkState() {
    if (state.projectId) {
        const workState = {
            projectId: state.projectId,
            currentIndex: state.currentIndex,
            timestamp: Date.now()
        };
        localStorage.setItem('sam3_work_state', JSON.stringify(workState));
    }
}

// 恢复工作状态
async function restoreWorkState() {
    const saved = localStorage.getItem('sam3_work_state');
    if (!saved) return;

    try {
        const workState = JSON.parse(saved);
        if (!workState.projectId) return;
        const success = await selectProject(workState.projectId);
        if (!success) {
            localStorage.removeItem('sam3_work_state');
            return;
        }
        await rescanProjectImages();
        if (state.images.length > 0) {
            const index = Math.min(
                Math.max(Number(workState.currentIndex) || 0, 0),
                state.images.length - 1
            );
            loadImage(index);
        }
    } catch (error) {
        localStorage.removeItem('sam3_work_state');
        console.error('恢复工作状态失败:', error);
        showToast('恢复失败', error.message, 'danger');
    }
}

// 重新扫描项目图片文件夹
async function rescanProjectImages({ notify = false } = {}) {
    if (!state.projectId) return false;
    if (state.dirty && !(await saveAnnotations(false))) return false;
    try {
        const data = await apiRequest(`/api/project/${state.projectId}`);
        if (!data.project.image_dir) return true;
        const scanData = await apiRequest(
            `/api/project/${state.projectId}/load_images`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_dir: data.project.image_dir })
            }
        );
        replaceImages(scanData.images);
        updateImageList();
        if (notify) {
            showToast('扫描完成', `找到 ${scanData.count} 张图片`);
        }
        return true;
    } catch (error) {
        console.error('扫描文件夹失败:', error);
        showToast('图片目录不可用', error.message, 'danger');
        return false;
    }
}

// 清除当前图片的所有标注
async function clearCurrentAnnotations() {
    if (state.annotations.length === 0) {
        showToast('提示', '当前没有标注');
        return;
    }
    if (!(await confirmAction({
        title: '清除当前标注',
        message: `确定要清除当前图片的 ${state.annotations.length} 个标注吗？`,
        confirmText: '清除',
        danger: true
    }))) return;
    state.annotations = [];
    recordAnnotationMutation({ autosave: false });
    if (await saveAnnotations(false)) {
        showToast('成功', '已清除并写入磁盘');
    } else {
        showToast('清除尚未保存', '修改仍保留在页面，可撤销或重试保存', 'warning');
    }
}

function initCanvas() {
    canvas = document.getElementById('annotationCanvas');
    ctx = canvas.getContext('2d');

    canvas.addEventListener('pointerdown', onMouseDown);
    canvas.addEventListener('pointermove', onMouseMove);
    canvas.addEventListener('pointerup', onMouseUp);
    canvas.addEventListener('pointercancel', cancelPointerInteraction);
    canvas.addEventListener('wheel', onWheel, { passive: false });
    canvas.addEventListener('dblclick', onDoubleClick);

    // 禁用右键菜单（用于右键拖动）
    canvas.addEventListener('contextmenu', onContextMenu);

    // 键盘快捷键
    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keyup', onKeyUp);

    // 初始化工具栏按钮状态
    initToolbarState();
}

function initToolbarState() {
    // 设置默认工具为 point
    const pointBtn = document.getElementById('toolPoint');
    if (pointBtn) pointBtn.classList.add('active');

    // 设置默认为正样本
    const positiveBtn = document.getElementById('labelPositive');
    if (positiveBtn) positiveBtn.classList.add('active');
    updateToolGuidance();
}

function initEventListeners() {
    // 置信度滑块
    applyAccessibleButtonNames();
    document.getElementById('confidenceSlider').addEventListener('input', (e) => {
        state.confidence = e.target.value / 100;
        document.getElementById('confidenceValue').textContent = state.confidence.toFixed(2);
    });

    const imageSearch = document.getElementById('imageSearch');
    imageSearch.addEventListener('input', event => {
        clearTimeout(imageSearchTimer);
        imageSearchTimer = setTimeout(() => filterImages(event.target.value), 120);
    });

    const imageList = document.getElementById('imageList');
    imageList.addEventListener('scroll', () => {
        if (imageListRafId) return;
        imageListRafId = requestAnimationFrame(() => {
            renderImageListWindow();
            imageListRafId = null;
        });
    });
    imageList.addEventListener('click', event => {
        const item = event.target.closest('.image-item');
        if (item) loadImage(Number(item.dataset.imageIndex));
    });

    const annotationList = document.getElementById('annotationList');
    annotationList.addEventListener('scroll', () => {
        if (annotationListRafId) return;
        annotationListRafId = requestAnimationFrame(() => {
            renderAnnotationListWindow();
            annotationListRafId = null;
        });
    });
    annotationList.addEventListener('click', event => {
        const item = event.target.closest('.annotation-item');
        if (!item) return;
        const id = item.dataset.annotationId;
        if (event.target.closest('.delete-annotation')) {
            deleteAnnotation(id);
        } else {
            selectAnnotation(id);
        }
    });

    // 文本提示回车
    document.getElementById('textPrompt').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') segmentByText();
    });
}

// ==================== 工具切换 ====================

function setTool(tool) {
    state.currentTool = tool;
    document.querySelectorAll('.toolbar-group .btn-tool').forEach(button => {
        if (button.id?.startsWith('tool')) button.classList.remove('active');
    });
    const toolButton = document.getElementById(
        'tool' + tool.charAt(0).toUpperCase() + tool.slice(1)
    );
    toolButton?.classList.add('active');
    updateToolGuidance();
    updateCursor();

    if (tool === 'text') {
        const rightPanel = document.getElementById('rightPanel');
        rightPanel.classList.remove('collapsed', 'auto-collapse');
        document.getElementById('rightExpandBtn').style.display = 'none';
        const prompt = document.getElementById('textPrompt');
        prompt.scrollIntoView({ block: 'nearest' });
        prompt.focus();
    }
    redraw();
}

function updateToolGuidance() {
    const positive = state.isPositive ? '正样本' : '负样本';
    const guidance = {
        point: `点击添加${positive}点；可连续添加，按“分割”执行`,
        box: `拖动绘制${positive}框；可组合多个框后执行分割`,
        text: '在右侧输入提示词，Enter 分割当前图片',
        polygon: '依次点击顶点；双击或 Enter 完成，Backspace 撤点',
        edit: '点击标注以选中；Delete 删除，⌘Z / Ctrl+Z 撤销'
    };
    const hint = document.getElementById('toolHint');
    if (hint) hint.textContent = guidance[state.currentTool] || '';
    const promptTool = state.currentTool === 'point' || state.currentTool === 'box';
    document.getElementById('labelGroup').hidden = !promptTool;
    document.getElementById('confidenceGroup').hidden = state.currentTool !== 'text';
    document.getElementById('segmentPromptButton').hidden = !promptTool;
    document.getElementById('clearPromptButton').hidden =
        !promptTool && state.currentTool !== 'polygon';
    const finishPolygonButton = document.getElementById('finishPolygonButton');
    finishPolygonButton.hidden = state.currentTool !== 'polygon';
    finishPolygonButton.disabled = state.tempPolygon.length < 3;
}

function setLabel(isPositive) {
    state.isPositive = isPositive;
    document.getElementById('labelPositive').classList.toggle('active', isPositive);
    document.getElementById('labelNegative').classList.toggle('active', !isPositive);
    updateToolGuidance();
}

// ==================== Canvas事件处理 ====================

// CSS 负责视觉缩放，Canvas 内部始终保持原图分辨率。
function getCanvasCoords(e) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: (e.clientX - rect.left) * canvas.width / rect.width,
        y: (e.clientY - rect.top) * canvas.height / rect.height
    };
}

function onMouseDown(e) {
    e.preventDefault();
    if (e.pointerId !== undefined) {
        canvas.setPointerCapture(e.pointerId);
    }

    // 右键长按拖动视图
    if (e.button === 2) {
        state.panning = true;
        state.panStart = { x: e.clientX, y: e.clientY };
        canvas.style.cursor = 'grabbing';
        return;
    }

    // 中键拖动 或 空格+左键拖动（备用方式）
    if (e.button === 1 || (e.button === 0 && state.spacePressed)) {
        state.panning = true;
        state.panStart = { x: e.clientX, y: e.clientY };
        canvas.style.cursor = 'grabbing';
        return;
    }

    // 左键操作
    if (e.button !== 0) return;

    const { x, y } = getCanvasCoords(e);

    if (state.currentTool === 'point') {
        addPoint(x, y);
    } else if (state.currentTool === 'box') {
        state.drawing = true;
        state.drawStart = { x, y };
    } else if (state.currentTool === 'edit') {
        selectAnnotationAt(x, y);
    } else if (state.currentTool === 'polygon') {
        addPolygonPoint(x, y);
    }
}

// 缓存的鼠标位置，用于节流
let lastMouseX = 0, lastMouseY = 0;

function onMouseMove(e) {
    // 平移拖动中
    if (state.panning && state.panStart) {
        const container = document.getElementById('canvasContainer');
        const dx = e.clientX - state.panStart.x;
        const dy = e.clientY - state.panStart.y;

        container.scrollLeft -= dx;
        container.scrollTop -= dy;

        state.panStart = { x: e.clientX, y: e.clientY };
        return;
    }

    // 框选绘制中
    if (!state.drawing) return;

    // 节流：如果鼠标移动距离太小，跳过绘制
    const moveDist = Math.abs(e.clientX - lastMouseX) + Math.abs(e.clientY - lastMouseY);
    if (moveDist < 2) return;  // 移动小于2像素时跳过
    lastMouseX = e.clientX;
    lastMouseY = e.clientY;

    const { x, y } = getCanvasCoords(e);

    if (state.currentTool === 'box' && state.drawStart) {
        // 使用 requestAnimationFrame 优化性能
        if (rafId) {
            cancelAnimationFrame(rafId);
        }
        const dynamicBox = {
            x1: Math.min(state.drawStart.x, x),
            y1: Math.min(state.drawStart.y, y),
            x2: Math.max(state.drawStart.x, x),
            y2: Math.max(state.drawStart.y, y)
        };
        rafId = requestAnimationFrame(() => {
            quickRedraw(dynamicBox);
            rafId = null;
        });
    }
}

function cancelPointerInteraction(e) {
    state.panning = false;
    state.panStart = null;
    state.drawing = false;
    state.drawStart = null;
    if (e.pointerId !== undefined && canvas.hasPointerCapture(e.pointerId)) {
        canvas.releasePointerCapture(e.pointerId);
    }
    updateCursor();
    redraw();
}

function onMouseUp(e) {
    // 结束平移拖动
    if (e.pointerId !== undefined && canvas.hasPointerCapture(e.pointerId)) {
        canvas.releasePointerCapture(e.pointerId);
    }
    if (state.panning) {
        state.panning = false;
        state.panStart = null;
        updateCursor();
        return;
    }

    if (!state.drawing) return;

    const { x, y } = getCanvasCoords(e);

    if (state.currentTool === 'box' && state.drawStart) {
        const box = normalizeBox(state.drawStart.x, state.drawStart.y, x, y);
        if (box.width > 5 && box.height > 5) {
            // 保存临时框，包含当前的正负样本标签
            box.label = state.isPositive;
            state.tempBoxes.push(box);
            redraw();
            const labelStr = state.isPositive ? '正样本' : '负样本';
            showToast('提示', `${labelStr}框已添加，可继续添加或点击"分割"按钮`);
        }
    }

    state.drawing = false;
    state.drawStart = null;
}

function onDoubleClick(e) {
    // 双击完成多边形绘制
    if (state.currentTool === 'polygon' && state.tempPolygon.length >= 3) {
        e.preventDefault();
        finishPolygon();
    }
}

// 禁用右键菜单
function onContextMenu(e) {
    e.preventDefault();
    return false;
}

// 缩放节流控制
let zoomRafId = null;
let pendingZoomData = null;

// 滚轮缩放 - 以鼠标位置为中心
function onWheel(e) {
    e.preventDefault();
    if (!currentImage) return;

    const container = document.getElementById('canvasContainer');
    const wrapper = document.getElementById('canvasWrapper');
    const canvasRect = canvas.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();

    // 鼠标在canvas上的位置
    const mouseXOnCanvas = e.clientX - canvasRect.left;
    const mouseYOnCanvas = e.clientY - canvasRect.top;

    // CSS 尺寸已按 zoom 缩放。
    const imgX = mouseXOnCanvas / state.zoom;
    const imgY = mouseYOnCanvas / state.zoom;

    // 鼠标在容器视口中的位置
    const mouseXInViewport = e.clientX - containerRect.left;
    const mouseYInViewport = e.clientY - containerRect.top;

    // 计算新的缩放比例
    const oldZoom = state.zoom;
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(0.1, Math.min(10, oldZoom * delta));

    if (newZoom === oldZoom) return;

    // 使用 requestAnimationFrame 节流缩放操作
    pendingZoomData = {
        newZoom,
        imgX,
        imgY,
        mouseXInViewport,
        mouseYInViewport,
        container,
        wrapper
    };

    if (!zoomRafId) {
        zoomRafId = requestAnimationFrame(() => {
            if (pendingZoomData) {
                const data = pendingZoomData;
                state.zoom = data.newZoom;
                applyCanvasZoom();

                // 获取wrapper的padding（使用计算样式）
                const wrapperStyle = window.getComputedStyle(data.wrapper);
                const paddingLeft = parseFloat(wrapperStyle.paddingLeft);
                const paddingTop = parseFloat(wrapperStyle.paddingTop);

                // 缩放后，该图像点在新canvas上的位置
                const newPointOnCanvasX = data.imgX * data.newZoom;
                const newPointOnCanvasY = data.imgY * data.newZoom;

                // 该点在wrapper中的位置 = padding + canvas上的位置
                const pointInWrapperX = paddingLeft + newPointOnCanvasX;
                const pointInWrapperY = paddingTop + newPointOnCanvasY;

                // 设置滚动位置，使该点保持在鼠标位置
                data.container.scrollLeft = pointInWrapperX - data.mouseXInViewport;
                data.container.scrollTop = pointInWrapperY - data.mouseYInViewport;

                // 更新缩放显示
                updateZoomDisplay();
                pendingZoomData = null;
            }
            zoomRafId = null;
        });
    }
}

// ==================== 面板折叠功能 ====================

// 切换左侧面板
function toggleLeftPanel() {
    const panel = document.getElementById('leftPanel');
    const expandBtn = document.getElementById('leftExpandBtn');

    // 如果有auto-collapse，先移除它
    const wasAutoCollapsed = panel.classList.contains('auto-collapse');
    if (wasAutoCollapsed) {
        panel.classList.remove('auto-collapse');
        expandBtn.style.display = 'none';
        return; // 只是取消自动折叠，不改变手动状态
    }

    panel.classList.toggle('collapsed');

    // 控制展开按钮显示
    if (panel.classList.contains('collapsed')) {
        expandBtn.style.display = 'flex';
    } else {
        expandBtn.style.display = 'none';
    }

    // 保存状态
    localStorage.setItem('leftPanelCollapsed', panel.classList.contains('collapsed'));
}

// 切换右侧面板
function toggleRightPanel() {
    const panel = document.getElementById('rightPanel');
    const expandBtn = document.getElementById('rightExpandBtn');

    // 如果有auto-collapse，先移除它
    const wasAutoCollapsed = panel.classList.contains('auto-collapse');
    if (wasAutoCollapsed) {
        panel.classList.remove('auto-collapse');
        expandBtn.style.display = 'none';
        return; // 只是取消自动折叠，不改变手动状态
    }

    panel.classList.toggle('collapsed');

    // 控制展开按钮显示
    if (panel.classList.contains('collapsed')) {
        expandBtn.style.display = 'flex';
    } else {
        expandBtn.style.display = 'none';
    }

    // 保存状态
    localStorage.setItem('rightPanelCollapsed', panel.classList.contains('collapsed'));
}

// 恢复面板状态
function restorePanelState() {
    const leftCollapsed = localStorage.getItem('leftPanelCollapsed') === 'true';
    const rightCollapsed = localStorage.getItem('rightPanelCollapsed') === 'true';

    if (leftCollapsed) {
        document.getElementById('leftPanel').classList.add('collapsed');
        document.getElementById('leftExpandBtn').style.display = 'flex';
    }

    if (rightCollapsed) {
        document.getElementById('rightPanel').classList.add('collapsed');
        document.getElementById('rightExpandBtn').style.display = 'flex';
    }
}

// 响应式自动折叠
function handleResponsiveCollapse() {
    const width = window.innerWidth;
    const leftPanel = document.getElementById('leftPanel');
    const rightPanel = document.getElementById('rightPanel');
    const leftExpandBtn = document.getElementById('leftExpandBtn');
    const rightExpandBtn = document.getElementById('rightExpandBtn');

    // 小于1100px时自动折叠右侧
    if (width < 1100) {
        if (!rightPanel.classList.contains('collapsed')) {
            rightPanel.classList.add('auto-collapse');
            rightExpandBtn.style.display = 'flex';
        }
    } else {
        rightPanel.classList.remove('auto-collapse');
        // 如果不是手动折叠的，隐藏展开按钮
        if (!rightPanel.classList.contains('collapsed')) {
            rightExpandBtn.style.display = 'none';
        }
    }

    // 小于900px时自动折叠左侧
    if (width < 900) {
        if (!leftPanel.classList.contains('collapsed')) {
            leftPanel.classList.add('auto-collapse');
            leftExpandBtn.style.display = 'flex';
        }
    } else {
        leftPanel.classList.remove('auto-collapse');
        // 如果不是手动折叠的，隐藏展开按钮
        if (!leftPanel.classList.contains('collapsed')) {
            leftExpandBtn.style.display = 'none';
        }
    }
}

// 监听窗口大小变化
window.addEventListener('resize', handleResponsiveCollapse);

// 更新缩放显示
function updateZoomDisplay() {
    const zoomBtn = document.querySelector('.canvas-toolbar .small-text');
    if (zoomBtn) {
        zoomBtn.textContent = Math.round(state.zoom * 100) + '%';
    }
}

// 更新光标样式
function updateCursor() {
    if (state.spacePressed) {
        canvas.style.cursor = 'grab';
    } else if (state.currentTool === 'edit' || state.currentTool === 'text') {
        canvas.style.cursor = 'default';
    } else {
        canvas.style.cursor = 'crosshair';
    }
}

function onKeyDown(e) {
    // 在输入框内保留系统文本编辑快捷键；Escape 仅退出输入。
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
        if (e.key === 'Escape') e.target.blur();
        return;
    }

    // ESC键 - 取消当前操作
    if (e.key === 'Escape') {
        e.preventDefault();
        // 优先取消多边形绘制
        if (state.tempPolygon.length > 0) {
            cancelPolygon();
            return;
        }
        // 如果有临时标记，先清除
        if (state.tempBoxes.length > 0 || state.tempPoints.length > 0) {
            clearTempPrompts();
            return;
        }
        // 取消选中
        if (state.selectedAnnotation) {
            state.selectedAnnotation = null;
            updateAnnotationList();
            redraw();
        }
        return;
    }

    // Enter键 - 完成多边形绘制
    if (e.key === 'Enter' && state.tempPolygon.length >= 3) {
        e.preventDefault();
        finishPolygon();
        return;
    }

    // Backspace键 - 撤销多边形最后一个点
    if (e.key === 'Backspace' && state.tempPolygon.length > 0) {
        e.preventDefault();
        undoPolygonPoint();
        return;
    }

    // 空格键 - 进入平移模式
    if (e.code === 'Space' && !state.spacePressed) {
        e.preventDefault();
        state.spacePressed = true;
        updateCursor();
        return;
    }

    const key = e.key.toLowerCase();
    const commandKey = e.ctrlKey || e.metaKey;
    if (commandKey && key === 'z') {
        e.preventDefault();
        if (e.shiftKey) redoAnnotationChange();
        else undoAnnotationChange();
    }
    else if (commandKey && key === 'y') {
        e.preventDefault();
        redoAnnotationChange();
    }
    else if (commandKey && key === 's') {
        e.preventDefault();
        saveAnnotations();
    }
    else if (e.key === 'ArrowLeft') prevImage();
    else if (e.key === 'ArrowRight') nextImage();
    else if (e.key === 'Delete' && state.selectedAnnotation) deleteSelectedAnnotation();
    else if (key === 'p') setTool('point');
    else if (key === 'b') setTool('box');
    else if (key === 't') setTool('text');
    else if (key === 'e') setTool('edit');
    else if (key === 'g') setTool('polygon');
}

function onKeyUp(e) {
    // 空格键释放 - 退出平移模式
    if (e.code === 'Space') {
        state.spacePressed = false;
        // 如果正在拖动，结束拖动
        if (state.panning) {
            state.panning = false;
            state.panStart = null;
        }
        updateCursor();
    }
}

// ==================== 绘制函数 ====================

// 标记静态缓存需要更新
function invalidateStaticCache() {
    staticCacheDirty = true;
}

// 更新源分辨率静态缓存（图像 + 标注）。缩放仅改变 CSS 尺寸。
function updateStaticCache() {
    if (!currentImage) return;

    if (!staticCanvas) {
        staticCanvas = document.createElement('canvas');
        staticCtx = staticCanvas.getContext('2d', { alpha: false });
    }
    if (
        staticCanvas.width !== currentImage.naturalWidth ||
        staticCanvas.height !== currentImage.naturalHeight
    ) {
        staticCanvas.width = currentImage.naturalWidth;
        staticCanvas.height = currentImage.naturalHeight;
    }

    staticCtx.imageSmoothingEnabled = true;
    staticCtx.imageSmoothingQuality = 'medium';
    staticCtx.clearRect(0, 0, staticCanvas.width, staticCanvas.height);
    staticCtx.drawImage(currentImage, 0, 0);

    const originalCtx = ctx;
    ctx = staticCtx;
    state.annotations.forEach((annotation, index) => {
        drawAnnotation(annotation, index);
    });
    ctx = originalCtx;
    staticCacheDirty = false;
}

function redraw() {
    if (!currentImage) return;

    if (
        canvas.width !== currentImage.naturalWidth ||
        canvas.height !== currentImage.naturalHeight
    ) {
        canvas.width = currentImage.naturalWidth;
        canvas.height = currentImage.naturalHeight;
        staticCacheDirty = true;
    }
    if (staticCacheDirty) {
        updateStaticCache();
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (staticCanvas) {
        ctx.drawImage(staticCanvas, 0, 0);
    }
    state.tempPoints.forEach(point => {
        drawPoint(point.x, point.y, point.label);
    });
    if (state.tempPolygon.length > 0) {
        drawTempPolygon();
    }
    state.tempBoxes.forEach(box => {
        drawTempBox(box.x1, box.y1, box.x2, box.y2, box.label);
    });
}

// 快速重绘：只绘制动态内容，用于拖动绘制时
function quickRedraw(dynamicBox) {
    if (!currentImage || !staticCanvas) {
        redraw();
        if (dynamicBox) {
            drawTempBox(dynamicBox.x1, dynamicBox.y1, dynamicBox.x2, dynamicBox.y2);
        }
        return;
    }

    // 从缓存绘制静态内容
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(staticCanvas, 0, 0);

    state.tempPoints.forEach(point => {
        drawPoint(point.x, point.y, point.label);
    });
    if (state.tempPolygon.length > 0) {
        drawTempPolygon();
    }

    // 绘制已保存的临时框
    state.tempBoxes.forEach(box => {
        drawTempBox(box.x1, box.y1, box.x2, box.y2, box.label);
    });

    // 绘制正在拖动的框
    if (dynamicBox) {
        drawTempBox(dynamicBox.x1, dynamicBox.y1, dynamicBox.x2, dynamicBox.y2);
    }
}

function drawAnnotation(ann, idx) {
    const color = colors[idx % colors.length];
    const isSelected = state.selectedAnnotation === ann.id;
    const number = idx + 1;
    const lineWidth = isSelected ? 3 : 2;

    // 绘制多边形
    if (ann.polygon && ann.polygon.length > 2) {
        ctx.beginPath();
        ctx.moveTo(ann.polygon[0][0], ann.polygon[0][1]);
        for (let i = 1; i < ann.polygon.length; i++) {
            ctx.lineTo(ann.polygon[i][0], ann.polygon[i][1]);
        }
        ctx.closePath();
        ctx.fillStyle = color + '40';
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.stroke();
    }

    // 绘制边界框和简洁标签
    if (ann.bbox) {
        const [x1, y1, x2, y2] = ann.bbox;
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // 简洁标签: car1, car2
        const label = ann.class_name || ann.label || 'obj';
        const displayText = `${label}${number}`;
        ctx.font = '12px sans-serif';
        const textWidth = ctx.measureText(displayText).width;

        // 小标签背景
        ctx.fillStyle = color;
        ctx.fillRect(x1, y1 - 18, textWidth + 8, 18);

        // 标签文字
        ctx.fillStyle = '#fff';
        ctx.fillText(displayText, x1 + 4, y1 - 5);
    }
}

function drawPoint(x, y, isPositive) {
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fillStyle = isPositive ? '#4ade80' : '#e94560';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();
}

function drawTempBox(x1, y1, x2, y2, label) {
    const isPositive = typeof label === 'boolean' ? label : state.isPositive;
    ctx.save();
    ctx.strokeStyle = isPositive ? '#4ade80' : '#e94560';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.restore();
}

function drawTempPolygon() {
    if (state.tempPolygon.length === 0) return;

    const points = state.tempPolygon;

    // 绘制多边形线条
    ctx.strokeStyle = '#fbbf24';  // 黄色
    ctx.fillStyle = 'rgba(251, 191, 36, 0.2)';
    ctx.lineWidth = 2;
    ctx.setLineDash([]);

    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i][0], points[i][1]);
    }

    // 如果超过2个点，闭合并填充
    if (points.length > 2) {
        ctx.closePath();
        ctx.fill();
    }
    ctx.stroke();

    // 绘制顶点
    points.forEach((p, idx) => {
        ctx.beginPath();
        ctx.arc(p[0], p[1], 5, 0, Math.PI * 2);
        ctx.fillStyle = idx === 0 ? '#e94560' : '#fbbf24';  // 第一个点红色
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
    });

    // 如果有多个点，显示提示连接到第一个点
    if (points.length > 2) {
        ctx.setLineDash([5, 5]);
        ctx.strokeStyle = 'rgba(251, 191, 36, 0.5)';
        ctx.beginPath();
        ctx.moveTo(points[points.length - 1][0], points[points.length - 1][1]);
        ctx.lineTo(points[0][0], points[0][1]);
        ctx.stroke();
        ctx.setLineDash([]);
    }
}

// ==================== 分割操作 ====================

function addPoint(x, y) {
    state.tempPoints.push({ x, y, label: state.isPositive });
    redraw();
    // 不自动分割，等待手动点击分割按钮
    showToast('提示', '点已添加，点击"分割"按钮执行分割');
}

// ==================== 手动多边形绘制 ====================

function addPolygonPoint(x, y) {
    state.tempPolygon.push([x, y]);
    updateToolGuidance();
    redraw();

    if (state.tempPolygon.length === 1) {
        showToast('提示', '继续点击添加顶点，双击或按Enter完成，ESC取消');
    }
}

function finishPolygon() {
    if (state.tempPolygon.length < 3) {
        showToast('提示', '多边形至少需要3个顶点');
        return;
    }

    // 检查类别
    const className = getCurrentClassName();
    if (!className) return;

    // 计算边界框
    const xs = state.tempPolygon.map(p => p[0]);
    const ys = state.tempPolygon.map(p => p[1]);
    const x1 = Math.min(...xs);
    const y1 = Math.min(...ys);
    const x2 = Math.max(...xs);
    const y2 = Math.max(...ys);

    // 计算面积 (Shoelace formula)
    let area = 0;
    for (let i = 0; i < state.tempPolygon.length; i++) {
        const j = (i + 1) % state.tempPolygon.length;
        area += state.tempPolygon[i][0] * state.tempPolygon[j][1];
        area -= state.tempPolygon[j][0] * state.tempPolygon[i][1];
    }
    area = Math.abs(area) / 2;

    // 创建标注
    const annotation = {
        id: generateId(),
        label: className,
        class_name: className,
        score: 1.0,  // 手动标注置信度为1
        bbox: [x1, y1, x2, y2],
        polygon: state.tempPolygon.slice(),  // 复制数组
        area: area,
        manual: true  // 标记为手动绘制
    };

    state.annotations.push(annotation);
    state.tempPolygon = [];
    updateToolGuidance();
    recordAnnotationMutation();
    showToast('已添加', `${className} · 正在自动保存`);
}

function cancelPolygon() {
    if (state.tempPolygon.length > 0) {
        state.tempPolygon = [];
        updateToolGuidance();
        redraw();
        showToast('提示', '已取消多边形绘制');
    }
}

function undoPolygonPoint() {
    if (state.tempPolygon.length > 0) {
        state.tempPolygon.pop();
        updateToolGuidance();
        redraw();
    }
}

function generateId() {
    return Math.random().toString(36).substr(2, 8);
}

// 手动触发分割（点击分割按钮）
async function segmentManual() {
    if (!state.projectId || state.currentIndex < 0) {
        showToast('提示', '请先选择项目和图片');
        return;
    }

    // 优先分割临时框
    if (state.tempBoxes.length > 0) {
        const applied = await segmentByBoxes(state.tempBoxes);
        if (applied) {
            state.tempBoxes = [];
            redraw();
        }
        return;
    }

    // 其次分割临时点
    if (state.tempPoints.length > 0) {
        await segmentByPoints();
        return;
    }

    showToast('提示', '请先绘制框或添加点');
}
// 清除临时提示。
function clearTempPrompts() {
    state.tempBoxes = [];
    state.tempPoints = [];
    state.tempPolygon = [];
    updateToolGuidance();
    redraw();
    showToast('提示', '已清除未完成的临时标记');
}

async function applySegmentationResults(results, className, emptyMessage, successMessage) {
    if (!Array.isArray(results) || results.length === 0) {
        showToast('未检测到对象', emptyMessage, 'warning');
        return false;
    }
    results.forEach(result => {
        result.class_name = className;
    });
    state.annotations.push(...results);
    recordAnnotationMutation({ autosave: false });
    const saved = await saveAnnotations(false);
    if (saved) {
        showToast('成功', successMessage);
    } else {
        showToast('分割完成但未保存', '结果仍保留在当前页面，请重试“保存”', 'warning');
    }
    return true;
}


async function segmentByPoints() {
    if (!state.projectId || state.currentIndex < 0) return false;
    const className = getCurrentClassName();
    if (!className) return false;

    const image = state.images[state.currentIndex];
    const points = state.tempPoints.map(point => [
        point.x,
        point.y,
        point.label ? 1 : 0
    ]);
    showLoading('正在分割...');
    try {
        const data = await apiRequest('/api/segment/point', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_path: image.path, points })
        });
        const applied = await applySegmentationResults(
            data.results,
            className,
            '请调整正负提示点后重试',
            `检测到 ${data.results.length} 个 \"${className}\"`
        );
        if (applied) {
            state.tempPoints = [];
            redraw();
        }
        return applied;
    } catch (error) {
        showToast('点击分割失败', error.message, 'danger');
        return false;
    } finally {
        hideLoading();
    }
}

async function segmentByBoxes(boxes) {
    if (!state.projectId || state.currentIndex < 0) return false;
    const className = getCurrentClassName();
    if (!className) return false;

    const image = state.images[state.currentIndex];
    const boxData = boxes.map(box => [
        box.x1,
        box.y1,
        box.x2,
        box.y2,
        box.label ? 1 : 0
    ]);
    showLoading('正在分割...');
    try {
        const data = await apiRequest('/api/segment/box', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_path: image.path, boxes: boxData })
        });
        return await applySegmentationResults(
            data.results,
            className,
            '请调整正负提示框后重试',
            `检测到 ${data.results.length} 个 \"${className}\"`
        );
    } catch (error) {
        showToast('框选分割失败', error.message, 'danger');
        return false;
    } finally {
        hideLoading();
    }
}

// 获取当前选中的类名
function getCurrentClassName() {
    if (state.classes.length === 0) {
        showToast('提示', '请先在类别管理中添加类名');
        return null;
    }

    // 如果有当前选中的类名，使用它
    if (state.currentClass && state.classes.includes(state.currentClass)) {
        return state.currentClass;
    }

    // 如果只有一个类名，直接使用
    if (state.classes.length === 1) {
        return state.classes[0];
    }

    // 多个类名，提示用户选择
    showToast('提示', '请先在类别管理中点击选择一个类名');
    return null;
}

async function segmentByText() {
    if (!state.projectId || state.currentIndex < 0) {
        console.log('[DEBUG] segmentByText: no project or image selected');
        showToast('提示', '请先选择项目和图片');
        return;
    }

    // 检查类别列表
    if (state.classes.length === 0) {
        showToast('提示', '请先在类别管理中添加类名');
        return;
    }

    const prompt = document.getElementById('textPrompt').value.trim();
    if (!prompt) {
        showToast('提示', '请输入分割提示词');
        return;
    }

    // 匹配类名：优先精确匹配，其次模糊匹配，否则使用当前选中的类名
    let matchedClass = findMatchingClass(prompt);

    if (!matchedClass) {
        // 没有匹配到，提示用户选择
        if (state.classes.length === 1) {
            matchedClass = state.classes[0];
        } else {
            showClassSelectModal(prompt);
            return;
        }
    }

    await executeTextSegment(prompt, matchedClass);
}

// 查找匹配的类名
function findMatchingClass(prompt) {
    const promptLower = prompt.toLowerCase();

    // 精确匹配
    for (const cls of state.classes) {
        if (cls.toLowerCase() === promptLower) {
            return cls;
        }
    }

    // 包含匹配（提示词包含类名，或类名包含提示词）
    for (const cls of state.classes) {
        const clsLower = cls.toLowerCase();
        if (promptLower.includes(clsLower) || clsLower.includes(promptLower)) {
            return cls;
        }
    }

    return null;
}

// 显示类名选择弹窗
function showClassSelectModal(prompt) {
    // 创建选择弹窗
    const modalHtml = `
        <div class="modal fade" id="classSelectModal" tabindex="-1">
            <div class="modal-dialog modal-sm">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">选择类别</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="关闭"></button>
                    </div>
                    <div class="modal-body">
                        <p class="small text-muted mb-3">提示词 "${escapeHtml(prompt)}" 未匹配到类别，请选择：</p>
                        <div class="class-select-list">
                            ${state.classes.map(cls => `
                                <button class="btn btn-outline-primary w-100 mb-2 class-select-btn"
                                        data-class="${escapeHtml(cls)}" data-prompt="${escapeHtml(prompt)}">
                                    ${escapeHtml(cls)}
                                </button>
                            `).join('')}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;

    // 移除旧的弹窗
    const oldModal = document.getElementById('classSelectModal');
    if (oldModal) oldModal.remove();

    // 添加新弹窗
    document.body.insertAdjacentHTML('beforeend', modalHtml);

    // 绑定点击事件
    document.querySelectorAll('.class-select-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const selectedClass = btn.dataset.class;
            const prompt = btn.dataset.prompt;
            bootstrap.Modal.getInstance(document.getElementById('classSelectModal')).hide();
            await executeTextSegment(prompt, selectedClass);
        });
    });

    // 显示弹窗
    new bootstrap.Modal(document.getElementById('classSelectModal')).show();
}

// 执行文本分割
async function executeTextSegment(prompt, className) {
    const image = state.images[state.currentIndex];
    let actualPrompt = prompt;
    let wasTranslated = false;

    try {
        if (aiConfig.enabled && aiConfig.apiUrl && aiConfig.apiKey) {
            showLoading(`正在翻译: ${prompt}...`);
            const translateResult = await translatePrompt(prompt);
            if (translateResult.translated) {
                actualPrompt = translateResult.text;
                wasTranslated = true;
            }
        }
        showLoading(
            `正在分割: ${wasTranslated ? `${actualPrompt} (${prompt})` : prompt}...`
        );
        const data = await apiRequest('/api/segment/text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_path: image.path,
                prompt: actualPrompt,
                confidence: state.confidence
            })
        });
        const suffix = wasTranslated ? `（翻译: ${actualPrompt}）` : '';
        await applySegmentationResults(
            data.results,
            className,
            `未检测到 \"${prompt}\"${suffix}`,
            `检测到 ${data.results.length} 个 \"${className}\"${suffix}`
        );
    } catch (error) {
        showToast('文本分割失败', error.message, 'danger');
    } finally {
        hideLoading();
    }
}

async function batchSegment() {
    if (!state.projectId || state.images.length === 0) {
        showToast('提示', '请先选择包含图片的项目', 'warning');
        return;
    }
    if (state.classes.length === 0) {
        showToast('提示', '请先在类别管理中添加类名', 'warning');
        return;
    }
    if (state.dirty && !(await saveAnnotations(false))) {
        showToast('无法开始批处理', '当前图片保存失败，请先重试保存', 'warning');
        return;
    }

    const prompt = document.getElementById('textPrompt').value.trim();
    if (!prompt) {
        showToast('提示', '请输入提示词', 'warning');
        return;
    }
    let className = findMatchingClass(prompt);
    if (!className) {
        className = state.classes.length === 1
            ? state.classes[0]
            : state.currentClass;
    }
    if (!className) {
        showToast('提示', `提示词 \"${prompt}\" 未匹配到类别，请先选择类别`, 'warning');
        return;
    }

    const startIndex = Math.max(parseInt(document.getElementById('batchStart').value) || 0, 0);
    const rawEnd = parseInt(document.getElementById('batchEnd').value);
    const endIndex = Number.isFinite(rawEnd) && rawEnd >= 0
        ? Math.min(rawEnd, state.images.length)
        : state.images.length;
    const skipAnnotated = document.getElementById('skipAnnotated').checked;
    const toProcess = [];
    for (let index = startIndex; index < endIndex; index++) {
        if (!skipAnnotated || !state.images[index].annotated) toProcess.push(index);
    }
    if (toProcess.length === 0) {
        showToast('提示', '没有需要处理的图片', 'warning');
        return;
    }

    let actualPrompt = prompt;
    let wasTranslated = false;
    if (aiConfig.enabled && aiConfig.apiUrl && aiConfig.apiKey) {
        showLoading(`正在翻译: ${prompt}...`);
        const translateResult = await translatePrompt(prompt);
        if (translateResult.translated) {
            actualPrompt = translateResult.text;
            wasTranslated = true;
        }
    }
    const translation = wasTranslated ? `\n翻译后: ${actualPrompt}` : '';
    if (!(await confirmAction({
        title: '开始批量分割',
        message: `即将处理 ${toProcess.length} 张图片\n类别: ${className}\n提示词: ${prompt}${translation}`,
        confirmText: '开始处理'
    }))) {
        hideLoading();
        return;
    }

    let processed = 0;
    let totalDetections = 0;
    const failures = [];
    showLoading(`正在批量分割... 0/${toProcess.length}`, 0);
    batchCancelRequested = false;
    const cancelButton = document.getElementById('cancelBatchButton');
    cancelButton.hidden = false;
    cancelButton.disabled = false;
    cancelButton.textContent = '完成当前图片后停止';
    try {
        for (let offset = 0; offset < toProcess.length; offset++) {
            if (batchCancelRequested) break;
            const imageIndex = toProcess[offset];
            const image = state.images[imageIndex];
            showLoading(
                `正在处理: ${image.filename} (${offset + 1}/${toProcess.length})`,
                (offset / toProcess.length) * 100
            );
            try {
                const current = await apiRequest(
                    `/api/annotation/get?project_id=${encodeURIComponent(state.projectId)}&image_index=${imageIndex}`
                );
                const segmented = await apiRequest('/api/segment/text', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image_path: image.path,
                        prompt: actualPrompt,
                        confidence: state.confidence
                    })
                });
                const newResults = segmented.results || [];
                newResults.forEach(result => {
                    result.class_name = className;
                });
                if (newResults.length > 0) {
                    const annotations = [...current.annotations, ...newResults];
                    await apiRequest('/api/annotation/save', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            project_id: state.projectId,
                            image_index: imageIndex,
                            annotations
                        })
                    });
                    setImageAnnotated(imageIndex, true);
                    if (imageIndex === state.currentIndex) {
                        state.annotations = annotations;
                    }
                }
                totalDetections += newResults.length;
                processed++;
            } catch (error) {
                failures.push(`${image.filename}: ${error.message}`);
            }
        }
    } finally {
        updateLoadingProgress(100);
        document.getElementById('cancelBatchButton').hidden = true;
        hideLoading();
    }

    const attempted = processed + failures.length;
    const canceled = attempted < toProcess.length;
    const summary = `${canceled ? '已停止' : '完成'} ${processed}/${toProcess.length} 张，新检测 ${totalDetections} 个对象`;
    if (failures.length > 0) {
        showToast(
            canceled ? '批量分割已停止' : '批量分割部分失败',
            `${summary}；失败 ${failures.length} 张：${failures.slice(0, 2).join('；')}`,
            'warning'
        );
    } else {
        showToast(canceled ? '批量分割已停止' : '批量分割完成', summary, canceled ? 'warning' : 'success');
    }
    updateImageList();
    if (state.currentIndex >= 0) loadImage(state.currentIndex);
}

// ==================== 项目管理 ====================

function showProjectModal() {
    loadProjects();
    new bootstrap.Modal(document.getElementById('projectModal')).show();
}

async function loadProjects() {
    const list = document.getElementById('projectList');
    try {
        const data = await apiRequest('/api/project/list');
        if (data.projects.length === 0) {
            list.innerHTML = '<div class="text-muted p-3">暂无项目</div>';
            return [];
        }

        list.innerHTML = data.projects.map(project => `
            <div class="list-group-item project-item" data-project-id="${escapeHtml(project.id)}">
                <div class="project-item-main">
                    <div class="d-flex justify-content-between align-items-center">
                        <strong>${escapeHtml(project.name)}</strong>
                        <small class="text-muted">${project.image_count ?? project.images?.length ?? 0} 张图片</small>
                    </div>
                    <small class="text-muted">${escapeHtml(project.image_dir || '未设置目录')}</small>
                </div>
                <div class="project-item-actions">
                    <button class="btn btn-sm btn-outline-primary edit-project" title="编辑">
                        <i class="bi bi-pencil"></i>
                    </button>
                    <button class="btn btn-sm btn-outline-danger delete-project"
                            data-project-name="${escapeHtml(project.name)}" title="删除">
                        <i class="bi bi-trash"></i>
                    </button>
                </div>
            </div>
        `).join('');
        applyAccessibleButtonNames(list);

        list.querySelectorAll('.project-item').forEach(item => {
            const projectId = item.dataset.projectId;
            item.querySelector('.project-item-main').addEventListener(
                'click',
                () => selectProject(projectId)
            );
            item.querySelector('.edit-project').addEventListener(
                'click',
                event => {
                    event.stopPropagation();
                    showEditProjectModal(projectId);
                }
            );
            item.querySelector('.delete-project').addEventListener(
                'click',
                event => {
                    event.stopPropagation();
                    deleteProject(projectId, event.currentTarget.dataset.projectName);
                }
            );
        });
        return data.projects;
    } catch (error) {
        list.innerHTML = '<div class="text-danger p-3">项目列表加载失败</div>';
        showToast('加载项目失败', error.message, 'danger');
        return [];
    }
}

// 编辑项目状态
let editingProjectId = null;

// 显示编辑项目模态框
async function showEditProjectModal(projectId) {
    try {
        const data = await apiRequest(`/api/project/${projectId}`);
        const project = data.project;
        editingProjectId = projectId;

        const tab = document.querySelector('a[href="#newProject"]');
        if (tab) new bootstrap.Tab(tab).show();
        document.getElementById('newProjectName').value = project.name || '';
        document.getElementById('newProjectImageDir').value = project.image_dir || '';
        document.getElementById('newProjectOutputDir').value = project.output_dir || '';
        document.getElementById('newProjectClasses').value =
            (project.classes || []).join(', ');

        const createBtn = document.querySelector('#newProject button.btn-primary');
        if (createBtn) {
            createBtn.innerHTML = '<i class="bi bi-check-lg"></i> 保存修改';
            createBtn.onclick = updateProject;
        }
        if (tab) tab.textContent = '编辑项目';
    } catch (error) {
        showToast('读取项目失败', error.message, 'danger');
    }
}

// 更新项目
async function updateProject() {
    if (!editingProjectId) {
        await createProject();
        return;
    }

    const name = document.getElementById('newProjectName').value.trim();
    const imageDir = document.getElementById('newProjectImageDir').value.trim();
    const outputDir = document.getElementById('newProjectOutputDir').value.trim();
    const classes = document.getElementById('newProjectClasses').value
        .split(',')
        .map(value => value.trim())
        .filter(Boolean);
    if (!name) {
        showToast('提示', '请输入项目名称', 'warning');
        return;
    }

    try {
        const projectId = editingProjectId;
        const data = await apiRequest(`/api/project/${projectId}/update`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name,
                image_dir: imageDir,
                output_dir: outputDir,
                classes
            })
        });

        if (state.projectId === projectId) {
            const directoryChanged = state.imageDir !== data.project.image_dir;
            state.imageDir = data.project.image_dir || '';
            state.outputDir = data.project.output_dir || '';
            state.classes = data.project.classes || [];
            document.getElementById('projectName').textContent = data.project.name;
            document.getElementById('exportOutputDir').value = state.outputDir;
            updateClassList();
            if (directoryChanged) {
                await rescanProjectImages({ notify: true });
                if (state.images.length > 0) await loadImage(0);
            }
        }

        resetProjectForm();
        await loadProjects();
        bootstrap.Modal.getInstance(document.getElementById('projectModal'))?.hide();
        showToast('成功', '项目已更新');
    } catch (error) {
        showToast('更新项目失败', error.message, 'danger');
    }
}

// 删除项目
async function deleteProject(projectId, projectName) {
    if (!(await confirmAction({
        title: '删除项目',
        message: `确定要删除项目 \"${projectName}\" 吗？\n此操作将删除项目及其所有标注数据，不可恢复。`,
        confirmText: '永久删除',
        danger: true
    }))) return;
    try {
        await apiRequest(`/api/project/${projectId}/delete`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        if (state.projectId === projectId) {
            state.projectId = null;
            state.imageDir = '';
            state.outputDir = '';
            replaceImages([]);
            state.annotations = [];
            state.classes = [];
            state.currentClass = null;
            state.currentIndex = 0;
            currentImage = null;
            document.getElementById('projectName').textContent = '未选择项目';
            updateImageList();
            updateClassList();
            updateAnnotationList();
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            localStorage.removeItem('sam3_work_state');
        }
        await loadProjects();
        showToast('成功', '项目已删除');
    } catch (error) {
        showToast('删除项目失败', error.message, 'danger');
    }
}

// 重置项目表单
function resetProjectForm() {
    editingProjectId = null;
    document.getElementById('newProjectName').value = '';
    document.getElementById('newProjectImageDir').value = '';
    document.getElementById('newProjectOutputDir').value = '';
    document.getElementById('newProjectClasses').value = '';

    // 恢复按钮文字
    const createBtn = document.querySelector('#newProject button.btn-primary');
    if (createBtn) {
        createBtn.innerHTML = '<i class="bi bi-plus-circle"></i> 创建项目';
        createBtn.onclick = createProject;
    }

    // 恢复标签页标题
    const tabLink = document.querySelector('a[href="#newProject"]');
    if (tabLink) {
        tabLink.textContent = '新建项目';
    }
}

// 监听模态框关闭事件，重置表单
document.addEventListener('DOMContentLoaded', () => {
    const projectModal = document.getElementById('projectModal');
    if (projectModal) {
        projectModal.addEventListener('hidden.bs.modal', resetProjectForm);
    }
});

async function createProject() {
    if (editingProjectId) {
        await updateProject();
        return;
    }

    const name = document.getElementById('newProjectName').value.trim();
    const imageDir = document.getElementById('newProjectImageDir').value.trim();
    const outputDir = document.getElementById('newProjectOutputDir').value.trim();
    const classes = document.getElementById('newProjectClasses').value
        .split(',')
        .map(value => value.trim())
        .filter(Boolean);
    if (!name) {
        showToast('提示', '请输入项目名称', 'warning');
        return;
    }

    try {
        const data = await apiRequest('/api/project/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name,
                image_dir: imageDir,
                output_dir: outputDir,
                classes
            })
        });
        await selectProject(data.project.id);
        if (data.project.image_dir) {
            await rescanProjectImages();
            if (state.images.length > 0) await loadImage(0);
        }
        bootstrap.Modal.getInstance(document.getElementById('projectModal'))?.hide();
        showToast(
            '项目创建成功',
            state.images.length > 0
                ? `已加载 ${state.images.length} 张图片`
                : '尚未加载图片，请编辑项目并选择图片目录'
        );
    } catch (error) {
        showToast('创建项目失败', error.message, 'danger');
    }
}

async function selectProject(projectId) {
    const projectToken = ++projectLoadToken;
    const navigationToken = ++imageLoadToken;
    cancelPreviewRequests();
    setImageLoading(false);
    updateCurrentImageInfo();
    if (state.dirty && !(await saveAnnotations(false))) {
        showToast('未切换项目', '当前标注尚未保存，请重试', 'warning');
        return false;
    }
    if (
        projectToken !== projectLoadToken
        || navigationToken !== imageLoadToken
    ) return false;

    try {
        const data = await apiRequest(`/api/project/${projectId}`);
        if (
            projectToken !== projectLoadToken
            || navigationToken !== imageLoadToken
        ) return false;
        const project = data.project;
        const images = project.images || [];
        const currentIndex = Math.min(
            Math.max(Number(project.current_index) || 0, 0),
            Math.max(images.length - 1, 0)
        );
        let loadedImage = null;
        let loadedAnnotations = [];
        if (images.length > 0) {
            setImageLoading(true, images[currentIndex], currentIndex, images.length);
            [loadedImage, loadedAnnotations] = await Promise.all([
                loadImageElement(images[currentIndex]),
                loadImageAnnotations(projectId, currentIndex)
            ]);
            if (
                projectToken !== projectLoadToken
                || navigationToken !== imageLoadToken
            ) return false;
        }

        state.projectId = projectId;
        state.imageDir = project.image_dir || '';
        state.outputDir = project.output_dir || '';
        state.classes = project.classes || [];
        replaceImages(images);
        state.currentIndex = currentIndex;
        state.annotations = loadedAnnotations;
        state.tempPoints = [];
        state.tempBoxes = [];
        state.tempPolygon = [];
        currentImage = loadedImage;
        resetAnnotationHistory();
        invalidateStaticCache();
        if (!state.classes.includes(state.currentClass)) {
            state.currentClass = state.classes[0] || null;
        }

        document.getElementById('projectName').textContent = project.name;
        document.getElementById('exportOutputDir').value = state.outputDir;
        updateClassList();
        updateAnnotationList();
        setImageLoading(false);
        updateCurrentImageInfo();
        updateImageList({ scrollCurrent: true });
        bootstrap.Modal.getInstance(document.getElementById('projectModal'))?.hide();

        if (currentImage) {
            fitToView();
            preloadAdjacentImages(currentIndex);
        } else {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
        saveWorkState();
        return true;
    } catch (error) {
        if (
            projectToken === projectLoadToken
            && navigationToken === imageLoadToken
        ) {
            setImageLoading(false);
            updateCurrentImageInfo();
            showToast('选择项目失败', error.message, 'danger');
        }
        return false;
    }
}

async function loadProjectImages(imageDir = state.imageDir) {
    if (!state.projectId || !imageDir) return false;
    if (state.dirty && !(await saveAnnotations(false))) return false;
    try {
        const data = await apiRequest(
            `/api/project/${state.projectId}/load_images`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_dir: imageDir })
            }
        );
        state.imageDir = data.image_dir;
        replaceImages(data.images);
        updateImageList();
        if (state.images.length > 0) {
            if (!(await loadImage(0))) return false;
            showToast('成功', `已加载 ${data.count} 张图片`);
        } else {
            showToast('图片目录为空', '没有找到支持的图片格式', 'warning');
        }
        return true;
    } catch (error) {
        showToast('加载图片失败', error.message, 'danger');
        return false;
    }
}

// ==================== 图片导航 ====================

function renderImageListWindow() {
    const list = document.getElementById('imageList');
    if (state.images.length === 0) {
        list.innerHTML = '<div class="empty-state"><i class="bi bi-images"></i><p>暂无图片</p></div>';
        return;
    }
    if (filteredImageIndices.length === 0) {
        list.innerHTML = '<div class="empty-state"><i class="bi bi-search"></i><p>没有匹配的图片</p></div>';
        return;
    }

    const viewportHeight = list.clientHeight || 480;
    const overscan = 8;
    const start = Math.max(
        0,
        Math.floor(list.scrollTop / IMAGE_ROW_HEIGHT) - overscan
    );
    const visibleCount = Math.ceil(viewportHeight / IMAGE_ROW_HEIGHT) + overscan * 2;
    const end = Math.min(filteredImageIndices.length, start + visibleCount);
    const rows = filteredImageIndices.slice(start, end).map(index => {
        const image = state.images[index];
        return `
            <div class="image-item ${index === state.currentIndex ? 'active' : ''} ${image.annotated ? 'annotated' : ''}"
                 data-image-index="${index}">
                <span class="index">${index + 1}</span>
                <span class="filename" title="${escapeHtml(image.filename)}">${escapeHtml(image.filename)}</span>
            </div>
        `;
    }).join('');
    const scrollTop = list.scrollTop;
    list.innerHTML = `
        <div class="image-list-spacer" style="height:${start * IMAGE_ROW_HEIGHT}px"></div>
        ${rows}
        <div class="image-list-spacer" style="height:${(filteredImageIndices.length - end) * IMAGE_ROW_HEIGHT}px"></div>
    `;
    list.scrollTop = scrollTop;
}

function updateImageList({ scrollCurrent = false } = {}) {
    const list = document.getElementById('imageList');
    const annotatedCount = state.annotatedCount;
    const total = state.images.length;
    const percent = total > 0 ? Math.round(annotatedCount / total * 100) : 0;
    document.getElementById('imageStats').textContent = `${annotatedCount}/${total}`;
    document.getElementById('progressBar').style.width = `${percent}%`;
    document.getElementById('progressText').textContent =
        `已标注: ${annotatedCount}/${total} (${percent}%)`;
    document.getElementById('prevImageButton').disabled =
        imageLoading || total === 0 || state.currentIndex <= 0;
    document.getElementById('nextImageButton').disabled =
        imageLoading || total === 0 || state.currentIndex >= total - 1;

    if (renderedImages !== state.images) {
        renderedImages = state.images;
        list.scrollTop = 0;
    }
    if (
        filteredImagesSource !== state.images
        || filteredImagesQuery !== imageFilterQuery
    ) {
        filteredImageIndices = [];
        for (let index = 0; index < state.images.length; index++) {
            if (state.images[index].filename.toLowerCase().includes(imageFilterQuery)) {
                filteredImageIndices.push(index);
            }
        }
        filteredImagesSource = state.images;
        filteredImagesQuery = imageFilterQuery;
    }

    if (scrollCurrent) {
        const position = imageFilterQuery
            ? filteredImageIndices.indexOf(state.currentIndex)
            : state.currentIndex;
        if (position >= 0) {
            const rowTop = position * IMAGE_ROW_HEIGHT;
            const rowBottom = rowTop + IMAGE_ROW_HEIGHT;
            if (rowTop < list.scrollTop || rowBottom > list.scrollTop + list.clientHeight) {
                list.scrollTop = Math.max(0, rowTop - list.clientHeight / 2);
            }
        }
    }
    renderImageListWindow();
}

function filterImages(query) {
    imageFilterQuery = query.trim().toLowerCase();
    document.getElementById('imageList').scrollTop = 0;
    updateImageList();
}

function loadImageElement(image) {
    return new Promise((resolve, reject) => {
        const nextImage = new Image();
        nextImage.onload = () => resolve(nextImage);
        nextImage.onerror = () => reject(
            new Error(`无法读取图片：${image.path}`)
        );
        nextImage.src = `/api/image/serve?path=${encodeURIComponent(image.path)}`;
    });
}

function loadImageAnnotations(projectId, imageIndex) {
    return apiRequest(
        `/api/annotation/get?project_id=${encodeURIComponent(projectId)}&image_index=${imageIndex}`
    ).then(data => data.annotations || []);
}

function setImageLoading(loading, image = null, index = 0, total = 0) {
    imageLoading = loading;
    const list = document.getElementById('imageList');
    list.setAttribute('aria-busy', String(loading));
    list.classList.toggle('is-loading', loading);
    if (loading && image) {
        document.getElementById('currentImageInfo').textContent =
            `正在加载 ${index + 1} / ${total} - ${image.filename}`;
    }
    const imageTotal = state.images.length;
    document.getElementById('prevImageButton').disabled =
        loading || imageTotal === 0 || state.currentIndex <= 0;
    document.getElementById('nextImageButton').disabled =
        loading || imageTotal === 0 || state.currentIndex >= imageTotal - 1;
}

function updateCurrentImageInfo() {
    const image = state.images[state.currentIndex];
    document.getElementById('currentImageInfo').textContent = image
        ? `${state.currentIndex + 1} / ${state.images.length} - ${image.filename}`
        : '未加载图片';
}

async function loadImage(index) {
    if (index < 0 || index >= state.images.length) return false;
    const loadToken = ++imageLoadToken;
    cancelPreviewRequests();
    setImageLoading(false);
    updateCurrentImageInfo();
    if (state.dirty && !(await saveAnnotations(false))) {
        if (loadToken === imageLoadToken) {
            showToast('未切换图片', '当前标注保存失败，请先解决保存问题', 'warning');
        }
        return false;
    }
    if (loadToken !== imageLoadToken) return false;

    const image = state.images[index];
    setImageLoading(true, image, index, state.images.length);
    try {
        const previousIndex = state.currentIndex;
        const [nextImage, annotations] = await Promise.all([
            loadImageElement(image),
            loadImageAnnotations(state.projectId, index)
        ]);
        if (loadToken !== imageLoadToken) return false;

        state.currentIndex = index;
        state.annotations = annotations;
        if (previousIndex !== index && state.images[previousIndex]) {
            delete state.images[previousIndex].annotations;
        }
        state.tempPoints = [];
        state.tempBoxes = [];
        state.tempPolygon = [];
        resetAnnotationHistory();
        currentImage = nextImage;
        invalidateStaticCache();
        setImageLoading(false);
        fitToView();
        updateAnnotationList();
        updateImageList({ scrollCurrent: true });
        updateCurrentImageInfo();
        saveWorkState();
        preloadAdjacentImages(index);
        return true;
    } catch (error) {
        if (loadToken === imageLoadToken) {
            setImageLoading(false);
            updateCurrentImageInfo();
            showToast('图片加载失败', error.message, 'danger');
        }
        return false;
    }
}

function preloadAdjacentImages(index) {
    [index - 1, index + 1].forEach(adjacentIndex => {
        const image = state.images[adjacentIndex];
        if (!image) return;
        const preloader = new Image();
        preloader.src = `/api/image/serve?path=${encodeURIComponent(image.path)}`;
    });
}

function prevImage() {
    if (state.currentIndex > 0) {
        loadImage(state.currentIndex - 1);
    }
}

function nextImage() {
    if (state.currentIndex < state.images.length - 1) {
        loadImage(state.currentIndex + 1);
    }
}

// ==================== 标注管理 ====================

function renderAnnotationListWindow({ scrollSelected = false } = {}) {
    const list = document.getElementById('annotationList');
    if (state.annotations.length === 0) {
        list.innerHTML = '<div class="text-muted small p-2">暂无标注</div>';
        return;
    }

    const imageKey = `${state.projectId || ''}:${state.currentIndex}`;
    if (renderedAnnotationImageKey !== imageKey) {
        renderedAnnotationImageKey = imageKey;
        list.scrollTop = 0;
    }
    if (scrollSelected && state.selectedAnnotation) {
        const selectedIndex = state.annotations.findIndex(
            annotation => annotation.id === state.selectedAnnotation
        );
        if (selectedIndex >= 0) {
            const rowTop = selectedIndex * ANNOTATION_ROW_HEIGHT;
            const rowBottom = rowTop + ANNOTATION_ROW_HEIGHT;
            if (
                rowTop < list.scrollTop
                || rowBottom > list.scrollTop + list.clientHeight
            ) {
                list.scrollTop = Math.max(0, rowTop - ANNOTATION_ROW_HEIGHT);
            }
        }
    }

    const viewportHeight = list.clientHeight || 250;
    const overscan = 5;
    const start = Math.max(
        0,
        Math.floor(list.scrollTop / ANNOTATION_ROW_HEIGHT) - overscan
    );
    const visibleCount = Math.ceil(
        viewportHeight / ANNOTATION_ROW_HEIGHT
    ) + overscan * 2;
    const end = Math.min(state.annotations.length, start + visibleCount);
    const rows = state.annotations.slice(start, end).map((annotation, offset) => {
        const index = start + offset;
        const id = escapeHtml(annotation.id);
        const label = escapeHtml(
            annotation.class_name || annotation.label || 'obj'
        );
        const score = Number(annotation.score) || 0;
        return `
            <div class="annotation-item ${state.selectedAnnotation === annotation.id ? 'selected' : ''}"
                 data-annotation-id="${id}">
                <div class="color-indicator" style="background-color: ${colors[index % colors.length]}"></div>
                <div class="info">
                    <div class="label">${label}${index + 1}</div>
                    <div class="score">置信度: ${score.toFixed(2)}</div>
                </div>
                <div class="actions">
                    <button class="btn btn-outline-danger btn-sm delete-annotation" title="删除标注">
                        <i class="bi bi-trash"></i>
                    </button>
                </div>
            </div>
        `;
    }).join('');
    const scrollTop = list.scrollTop;
    list.innerHTML = `
        <div style="height:${start * ANNOTATION_ROW_HEIGHT}px"></div>
        ${rows}
        <div style="height:${(state.annotations.length - end) * ANNOTATION_ROW_HEIGHT}px"></div>
    `;
    list.scrollTop = scrollTop;
    applyAccessibleButtonNames(list);
}

function updateAnnotationList(options = {}) {
    renderAnnotationListWindow(options);
}

function selectAnnotation(id) {
    state.selectedAnnotation = state.selectedAnnotation === id ? null : id;
    updateAnnotationList({ scrollSelected: true });
    redraw();
}

function selectAnnotationAt(x, y) {
    for (let i = state.annotations.length - 1; i >= 0; i--) {
        const ann = state.annotations[i];
        if (ann.bbox) {
            const [x1, y1, x2, y2] = ann.bbox;
            if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
                selectAnnotation(ann.id);
                return;
            }
        }
    }
    state.selectedAnnotation = null;
    updateAnnotationList();
    redraw();
}

async function deleteAnnotation(id, event) {
    if (event) event.stopPropagation();
    const nextAnnotations = state.annotations.filter(annotation => annotation.id !== id);
    if (nextAnnotations.length === state.annotations.length) return;
    state.annotations = nextAnnotations;
    recordAnnotationMutation({ autosave: false });
    if (!(await saveAnnotations(false))) {
        showToast('删除尚未保存', '标注仍保留在页面，可撤销或重试保存', 'warning');
    }
}

function deleteSelectedAnnotation() {
    if (state.selectedAnnotation) {
        deleteAnnotation(state.selectedAnnotation);
    }
}

function scheduleAnnotationAutosave() {
    clearTimeout(annotationAutosaveTimer);
    annotationAutosaveTimer = setTimeout(() => {
        annotationAutosaveTimer = null;
        saveAnnotations(false);
    }, 700);
}

async function saveAnnotations(showMessage = true) {
    if (!state.projectId || state.currentIndex < 0 || !state.images[state.currentIndex]) {
        return false;
    }
    if (!state.dirty) {
        if (showMessage) showToast('已保存', '当前标注没有未保存修改');
        return true;
    }

    clearTimeout(annotationAutosaveTimer);
    annotationAutosaveTimer = null;
    const projectId = state.projectId;
    const imageIndex = state.currentIndex;
    const revision = state.revision;
    const annotationCount = state.annotations.length;
    const payload = JSON.stringify({
        project_id: projectId,
        image_index: imageIndex,
        annotations: state.annotations
    });
    const saveKey = `${projectId}:${imageIndex}:${revision}`;
    if (lastQueuedSave?.key === saveKey) {
        return lastQueuedSave.promise;
    }
    updateSaveIndicator('saving');

    const execute = async () => {
        try {
            await apiRequest('/api/annotation/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: payload
            });
            if (state.projectId === projectId && state.images[imageIndex]) {
                setImageAnnotated(imageIndex, annotationCount > 0);
                updateImageList();
            }
            if (
                state.projectId === projectId &&
                state.currentIndex === imageIndex &&
                state.revision === revision
            ) {
                state.dirty = false;
                updateSaveIndicator('saved');
            } else if (state.projectId === projectId && state.currentIndex === imageIndex) {
                updateSaveIndicator('dirty');
            }
            if (showMessage) showToast('成功', '标注已安全写入磁盘');
            return true;
        } catch (error) {
            if (state.projectId === projectId && state.currentIndex === imageIndex) {
                updateSaveIndicator('error', '保存失败，修改仍在本地');
            }
            showToast('保存失败', error.message, 'danger');
            return false;
        }
    };

    const promise = saveQueue.then(execute, execute);
    saveQueue = promise.then(() => undefined, () => undefined);
    lastQueuedSave = { key: saveKey, promise };
    promise.finally(() => {
        if (lastQueuedSave?.key === saveKey) lastQueuedSave = null;
    });
    return promise;
}

// ==================== 类别管理 ====================

function updateClassList() {
    const list = document.getElementById('classList');
    if (state.classes.length === 0) {
        list.innerHTML = '<div class="text-muted small">暂无类别，请先添加</div>';
        state.currentClass = null;
        return;
    }
    if (!state.currentClass || !state.classes.includes(state.currentClass)) {
        state.currentClass = state.classes[0];
    }
    list.innerHTML = state.classes.map((className, index) => `
        <div class="class-item ${state.currentClass === className ? 'selected' : ''}"
             data-class-name="${escapeHtml(className)}" title="点击选择此类别">
            <div class="color-dot" style="background-color: ${colors[index % colors.length]}"></div>
            <span class="name">${escapeHtml(className)}</span>
            <button type="button" class="bi bi-x delete-btn" title="删除类别"
                    aria-label="删除类别"></button>
        </div>
    `).join('');
    list.querySelectorAll('.class-item').forEach(item => {
        const className = item.dataset.className;
        item.addEventListener('click', () => selectClass(className));
        item.querySelector('.delete-btn').addEventListener('click', event => {
            event.stopPropagation();
            removeClass(className);
        });
    });
}

// 选择当前类名
function selectClass(className) {
    state.currentClass = className;
    updateClassList();
    showToast('提示', `已选择类别: ${className}`);
}

async function addClass() {
    const input = document.getElementById('newClassName');
    const name = input.value.trim();
    if (!name || state.classes.includes(name)) return;

    const nextClasses = [...state.classes, name];
    try {
        if (state.projectId) {
            await apiRequest('/api/classes/update', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    project_id: state.projectId,
                    classes: nextClasses
                })
            });
        }
        state.classes = nextClasses;
        state.currentClass = name;
        updateClassList();
        input.value = '';
    } catch (error) {
        showToast('添加类别失败', error.message, 'danger');
    }
}

async function removeClass(name) {
    const used = state.annotations.some(
        annotation => (annotation.class_name || annotation.label) === name
    );
    if (used && !(await confirmAction({
        title: '删除正在使用的类别',
        message: `当前图片包含类别 \"${name}\" 的标注，仍要从类别列表删除吗？`,
        confirmText: '删除类别',
        danger: true
    }))) return;
    const previousClasses = state.classes;
    const nextClasses = state.classes.filter(className => className !== name);
    try {
        if (state.projectId) {
            await apiRequest('/api/classes/update', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    project_id: state.projectId,
                    classes: nextClasses
                })
            });
        }
        state.classes = nextClasses;
        updateClassList();
    } catch (error) {
        state.classes = previousClasses;
        updateClassList();
        showToast('删除类别失败', error.message, 'danger');
    }
}

// ==================== 导出 ====================

function showExportModal() {
    if (!state.projectId) {
        showToast('提示', '请先选择项目');
        return;
    }
    new bootstrap.Modal(document.getElementById('exportModal')).show();

    // 自动开始预览
    setTimeout(() => {
        generateExportPreview();
    }, 300);
}

async function exportDataset() {
    const format = document.getElementById('exportFormat').value;
    const outputDir = document.getElementById('exportOutputDir').value.trim();
    const smoothLevel = document.getElementById('exportSmoothLevel').value;
    const exportType = document.getElementById('exportType').value;
    if (!outputDir) {
        showToast('提示', '请选择输出目录', 'warning');
        return;
    }

    showLoading('正在导出...');
    try {
        const data = await apiRequest(`/api/export/${format}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project_id: state.projectId,
                output_dir: outputDir,
                smooth_level: smoothLevel,
                export_type: exportType
            })
        });
        const result = data.result;
        const smoothNames = {
            none: '无',
            low: '低',
            medium: '中等',
            high: '高',
            ultra: '超高'
        };
        showToast(
            '导出完成',
            `Train: ${result.train}, Val: ${result.val}, Test: ${result.test}\n` +
            `总标注: ${result.total_annotations}\n平滑级别: ${smoothNames[smoothLevel]}`
        );
        bootstrap.Modal.getInstance(document.getElementById('exportModal'))?.hide();
    } catch (error) {
        showToast('导出失败', error.message, 'danger');
    } finally {
        hideLoading();
    }
}

// ==================== 导出预览 ====================

async function generateExportPreview() {
    if (!state.projectId) {
        showToast('提示', '请先选择项目');
        return;
    }

    exportPreviewController?.abort();
    const controller = new AbortController();
    exportPreviewController = controller;
    const projectId = state.projectId;
    const imageIndex = state.currentIndex;
    const smoothLevel = document.getElementById('exportSmoothLevel').value;
    const previewImage = document.getElementById('exportPreviewImage');
    const placeholder = document.querySelector('.preview-placeholder');
    const statsDiv = document.getElementById('exportPreviewStats');

    if (placeholder) {
        placeholder.innerHTML = '<div class="spinner-border spinner-border-sm"></div><p>生成预览中...</p>';
        placeholder.style.display = 'block';
    }

    try {
        const data = await imageRequest('/api/export/preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: controller.signal,
            body: JSON.stringify({
                project_id: projectId,
                image_index: imageIndex,
                smooth_level: smoothLevel,
                show_polygon: true,
                show_fill: true,
                opacity: 0.4
            })
        });
        if (
            controller !== exportPreviewController
            || projectId !== state.projectId
            || imageIndex !== state.currentIndex
        ) return;
        releaseExportPreviewObjectUrl();
        exportPreviewObjectUrl = URL.createObjectURL(data.blob);
        previewImage.src = exportPreviewObjectUrl;
        previewImage.style.display = 'block';
        if (placeholder) placeholder.style.display = 'none';

        const stats = data.stats;
        const smoothNames = {
            none: '无平滑',
            low: '低',
            medium: '中等',
            high: '高',
            ultra: '超高'
        };
        statsDiv.textContent =
            `文件: ${stats.filename} | 标注数: ${stats.total_annotations} | ` +
            `平滑: ${smoothNames[stats.smooth_level] || '未知'} | ` +
            `尺寸: ${stats.image_size[0]}x${stats.image_size[1]}`;
        statsDiv.style.display = 'block';
    } catch (error) {
        if (error.name === 'AbortError') return;
        showToast('错误', error.message, 'danger');
        if (placeholder) {
            placeholder.innerHTML = '<i class="bi bi-exclamation-triangle"></i><p>预览生成失败</p>';
            placeholder.style.display = 'block';
        }
    } finally {
        if (controller === exportPreviewController) {
            exportPreviewController = null;
        }
    }
}

function updateExportPreview() {
    // 如果预览图片已显示，则自动更新预览
    const previewImage = document.getElementById('exportPreviewImage');
    if (previewImage && previewImage.style.display !== 'none') {
        generateExportPreview();
    }
}


// ==================== 图片放大查看器 ====================

let viewerZoom = 1;

function openImageViewer(imgSrc, info = '') {
    const viewer = document.getElementById('imageViewer');
    const viewerImg = document.getElementById('imageViewerImg');
    const viewerInfo = document.getElementById('imageViewerInfo');

    viewerImg.src = imgSrc;
    viewerInfo.textContent = info;
    viewerZoom = 1;
    viewerImg.style.transform = `scale(${viewerZoom})`;

    viewer.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeImageViewer(event) {
    // 如果点击的是背景或关闭按钮，则关闭
    if (!event || event.target.id === 'imageViewer' || event.target.classList.contains('image-viewer-close')) {
        const viewer = document.getElementById('imageViewer');
        viewer.classList.remove('active');
        document.body.style.overflow = '';
    }
}

function zoomViewerImage(factor) {
    const viewerImg = document.getElementById('imageViewerImg');
    viewerZoom *= factor;
    viewerZoom = Math.max(0.2, Math.min(5, viewerZoom)); // 限制缩放范围
    viewerImg.style.transform = `scale(${viewerZoom})`;
}

function resetViewerZoom() {
    const viewerImg = document.getElementById('imageViewerImg');
    viewerZoom = 1;
    viewerImg.style.transform = `scale(${viewerZoom})`;
}

// 为预览图添加点击事件
function initImageViewerEvents() {
    // 导出预览图
    const exportPreview = document.getElementById('exportPreviewImage');
    if (exportPreview) {
        exportPreview.onclick = function() {
            if (this.src && this.style.display !== 'none') {
                openImageViewer(this.src, '导出预览');
            }
        };
    }

}

// ESC 键关闭查看器
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        closeImageViewer();
    }
});

// 鼠标滚轮缩放
document.getElementById('imageViewer')?.addEventListener('wheel', function(e) {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    zoomViewerImage(factor);
});

// 页面加载后初始化
document.addEventListener('DOMContentLoaded', function() {
    initImageViewerEvents();
});

// ==================== 缩放控制 ====================

function applyCanvasZoom() {
    if (!currentImage) return;
    canvas.style.width = `${currentImage.naturalWidth * state.zoom}px`;
    canvas.style.height = `${currentImage.naturalHeight * state.zoom}px`;
    updateZoomDisplay();
}

function zoomIn() {
    state.zoom = Math.min(10, state.zoom * 1.2);
    applyCanvasZoom();
}

function zoomOut() {
    state.zoom = Math.max(0.1, state.zoom / 1.2);
    applyCanvasZoom();
}

function resetZoom() {
    state.zoom = 1;
    applyCanvasZoom();
    centerCanvas();
}

// 适应视图 - 让图片完整显示在容器中
function fitToView() {
    if (!currentImage) return;

    const container = document.getElementById('canvasContainer');
    const containerWidth = container.clientWidth - 40; // 留一些边距
    const containerHeight = container.clientHeight - 40;

    const scaleX = containerWidth / currentImage.width;
    const scaleY = containerHeight / currentImage.height;

    state.zoom = Math.min(scaleX, scaleY, 1);
    redraw();
    applyCanvasZoom();
    centerCanvas();
}

// 将 canvas 居中显示
function centerCanvas() {
    const container = document.getElementById('canvasContainer');
    const wrapper = document.getElementById('canvasWrapper');

    if (!wrapper || !container) return;

    // 计算居中位置
    const scrollX = (wrapper.scrollWidth - container.clientWidth) / 2;
    const scrollY = (wrapper.scrollHeight - container.clientHeight) / 2;

    container.scrollLeft = Math.max(0, scrollX);
    container.scrollTop = Math.max(0, scrollY);
}

// ==================== 工具函数 ====================

function normalizeBox(x1, y1, x2, y2) {
    return {
        x1: Math.min(x1, x2),
        y1: Math.min(y1, y2),
        x2: Math.max(x1, x2),
        y2: Math.max(y1, y2),
        width: Math.abs(x2 - x1),
        height: Math.abs(y2 - y1)
    };
}

function showLoading(text = '处理中...', progress = -1) {
    document.getElementById('loadingText').textContent = text;
    document.getElementById('loadingOverlay').style.display = 'flex';
    updateLoadingProgress(progress);
}

function updateLoadingProgress(progress) {
    const progressBar = document.getElementById('loadingProgress');
    const progressText = document.getElementById('loadingProgressText');

    if (progress < 0) {
        // 不确定进度，显示动画
        progressBar.style.width = '100%';
        progressBar.style.animation = 'loading-pulse 1.5s ease-in-out infinite';
        progressText.textContent = '处理中...';
    } else {
        // 确定进度
        progressBar.style.animation = 'none';
        progressBar.style.width = Math.min(100, Math.max(0, progress)) + '%';
        progressText.textContent = Math.round(progress) + '%';
    }
}

function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
    // 重置进度
    updateLoadingProgress(0);
}

function showToast(title, message, type = 'success') {
    const toast = document.getElementById('toast');
    document.getElementById('toastTitle').textContent = title;
    document.getElementById('toastBody').textContent = message;

    toast.classList.remove('bg-success', 'bg-danger', 'bg-warning');
    const background = {
        success: 'bg-success',
        danger: 'bg-danger',
        warning: 'bg-warning'
    }[type];
    if (background) toast.classList.add(background);

    new bootstrap.Toast(toast).show();
}

// 切换批量选项显示/隐藏
function toggleBatchOptions() {
    const options = document.getElementById('batchOptions');
    const toggle = document.querySelector('.batch-toggle');
    const icon = document.getElementById('batchToggleIcon');

    const isHidden = options.style.display === 'none';
    options.style.display = isHidden ? 'block' : 'none';
    toggle.classList.toggle('expanded', isHidden);
}

// ==================== AI翻译配置 ====================

// AI配置状态
const aiConfig = {
    enabled: false,
    apiUrl: '',
    apiKey: '',
    model: 'deepseek-chat'
};

// 初始化时加载AI配置
function loadAIConfig() {
    const saved = localStorage.getItem('sam3_ai_config');
    if (!saved) {
        aiConfig.apiKey = sessionStorage.getItem('sam3_ai_api_key') || '';
        updateAIConfigUI();
        return;
    }
    try {
        const config = JSON.parse(saved);
        aiConfig.enabled = Boolean(config.enabled);
        aiConfig.apiUrl = String(config.apiUrl || '');
        aiConfig.model = String(config.model || 'deepseek-chat');
        aiConfig.apiKey =
            sessionStorage.getItem('sam3_ai_api_key')
            || String(config.apiKey || '');
        if (config.apiKey) {
            sessionStorage.setItem('sam3_ai_api_key', aiConfig.apiKey);
            localStorage.setItem('sam3_ai_config', JSON.stringify({
                enabled: aiConfig.enabled,
                apiUrl: aiConfig.apiUrl,
                model: aiConfig.model
            }));
        }
        updateAIConfigUI();
    } catch (error) {
        console.error('加载AI配置失败:', error);
        localStorage.removeItem('sam3_ai_config');
    }
}

// 更新AI配置UI状态
function updateAIConfigUI() {
    const btn = document.getElementById('aiConfigBtn');
    const statusText = document.getElementById('aiStatusText');
    const enabledCheckbox = document.getElementById('aiTranslateEnabled');

    if (btn) {
        if (aiConfig.enabled && aiConfig.apiUrl && aiConfig.apiKey) {
            btn.classList.add('ai-enabled');
            btn.title = 'AI翻译已启用';
        } else {
            btn.classList.remove('ai-enabled');
            btn.title = 'AI翻译配置';
        }
    }

    if (statusText) {
        if (aiConfig.enabled && aiConfig.apiUrl && aiConfig.apiKey) {
            statusText.textContent = '已启用';
            statusText.style.color = 'var(--success)';
        } else if (aiConfig.apiUrl && !aiConfig.apiKey) {
            statusText.textContent = '需输入本次会话密钥';
            statusText.style.color = 'var(--warning)';
        } else if (aiConfig.apiUrl && aiConfig.apiKey) {
            statusText.textContent = '已配置但未启用';
            statusText.style.color = 'var(--warning)';
        } else {
            statusText.textContent = '未配置';
            statusText.style.color = 'var(--text-muted)';
        }
    }

    if (enabledCheckbox) {
        enabledCheckbox.checked = aiConfig.enabled;
    }
}

// 显示AI配置模态框
function showAIConfigModal() {
    // 填充当前配置
    document.getElementById('aiApiUrl').value = aiConfig.apiUrl || '';
    document.getElementById('aiApiKey').value = aiConfig.apiKey || '';
    document.getElementById('aiModel').value = aiConfig.model || 'deepseek-chat';
    document.getElementById('aiTranslateEnabled').checked = aiConfig.enabled;

    // 隐藏测试结果
    document.getElementById('aiTestResult').style.display = 'none';

    updateAIConfigUI();
    new bootstrap.Modal(document.getElementById('aiConfigModal')).show();
}

// 切换API密钥可见性
function toggleApiKeyVisibility() {
    const input = document.getElementById('aiApiKey');
    const icon = document.getElementById('apiKeyEyeIcon');

    if (input.type === 'password') {
        input.type = 'text';
        icon.classList.remove('bi-eye');
        icon.classList.add('bi-eye-slash');
    } else {
        input.type = 'password';
        icon.classList.remove('bi-eye-slash');
        icon.classList.add('bi-eye');
    }
}

// 测试AI配置
async function testAIConfig() {
    const apiUrl = document.getElementById('aiApiUrl').value.trim();
    const apiKey = document.getElementById('aiApiKey').value.trim();
    const model = document.getElementById('aiModel').value.trim();

    const resultDiv = document.getElementById('aiTestResult');

    if (!apiUrl || !apiKey) {
        resultDiv.innerHTML = '<div class="alert alert-warning small mb-0"><i class="bi bi-exclamation-triangle"></i> 请填写API地址和密钥</div>';
        resultDiv.style.display = 'block';
        return;
    }

    resultDiv.innerHTML = '<div class="alert alert-info small mb-0"><i class="bi bi-hourglass-split"></i> 正在测试连接...</div>';
    resultDiv.style.display = 'block';

    try {
        await apiRequest('/api/ai/test', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ api_url: apiUrl, api_key: apiKey, model })
        });
        resultDiv.innerHTML =
            '<div class="alert alert-success small mb-0">' +
            '<i class="bi bi-check-circle"></i> 连接成功！API配置有效</div>';
    } catch (error) {
        resultDiv.innerHTML = `<div class="alert alert-danger small mb-0"><i class="bi bi-x-circle"></i> 请求错误: ${escapeHtml(error.message)}</div>`;
    }
}

// 保存AI配置
function saveAIConfig() {
    aiConfig.apiUrl = document.getElementById('aiApiUrl').value.trim();
    aiConfig.apiKey = document.getElementById('aiApiKey').value.trim();
    aiConfig.model = document.getElementById('aiModel').value.trim() || 'deepseek-chat';
    aiConfig.enabled = document.getElementById('aiTranslateEnabled').checked;

    // 如果启用但未配置，提示用户
    if (aiConfig.enabled && (!aiConfig.apiUrl || !aiConfig.apiKey)) {
        showToast('提示', '请先配置API地址和密钥', 'warning');
        return;
    }

    sessionStorage.setItem('sam3_ai_api_key', aiConfig.apiKey);
    localStorage.setItem('sam3_ai_config', JSON.stringify({
        enabled: aiConfig.enabled,
        apiUrl: aiConfig.apiUrl,
        model: aiConfig.model
    }));
    updateAIConfigUI();

    bootstrap.Modal.getInstance(document.getElementById('aiConfigModal')).hide();

    if (aiConfig.enabled) {
        showToast('成功', 'AI翻译已启用，中文提示词将自动翻译');
    } else {
        showToast('成功', 'AI配置已保存');
    }
}

// 清除AI配置
async function clearAIConfig() {
    if (!(await confirmAction({
        title: '清除 AI 配置',
        message: '确定要清除保存的 API 地址、会话密钥和模型配置吗？',
        confirmText: '清除',
        danger: true
    }))) return;

    aiConfig.enabled = false;
    aiConfig.apiUrl = '';
    aiConfig.apiKey = '';
    aiConfig.model = 'deepseek-chat';
    localStorage.removeItem('sam3_ai_config');
    sessionStorage.removeItem('sam3_ai_api_key');

    document.getElementById('aiApiUrl').value = '';
    document.getElementById('aiApiKey').value = '';
    document.getElementById('aiModel').value = 'deepseek-chat';
    document.getElementById('aiTranslateEnabled').checked = false;
    document.getElementById('aiTestResult').style.display = 'none';
    updateAIConfigUI();
    showToast('成功', 'AI配置已清除');
}

// 翻译文本（如果启用了AI翻译）
async function translatePrompt(text) {
    // 如果未启用或未配置，直接返回原文
    if (!aiConfig.enabled || !aiConfig.apiUrl || !aiConfig.apiKey) {
        return { success: false, text: text };
    }

    // 检测是否包含中文
    const hasChinese = /[\u4e00-\u9fa5]/.test(text);
    if (!hasChinese) {
        return { success: true, text: text, translated: false };
    }

    try {
        const data = await apiRequest('/api/ai/translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text,
                api_url: aiConfig.apiUrl,
                api_key: aiConfig.apiKey,
                model: aiConfig.model
            })
        });

        console.log(`[AI翻译] "${text}" -> "${data.translated}"`);
        return { success: true, text: data.translated, original: text, translated: true };
    } catch (error) {
        console.error('[AI翻译错误]', error);
        return { success: false, text: text, error: error.message };
    }
}

// 在页面加载时初始化AI配置
document.addEventListener('DOMContentLoaded', () => {
    loadAIConfig();
});
