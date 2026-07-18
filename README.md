# SAM3 AN - 智能数据标注工具

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Flask-2.3+-green.svg" alt="Flask">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

基于 **SAM3 (Segment Anything Model 3)** 的智能数据标注工具，支持图像分割标注。通过文本提示、点击、框选等多种方式快速生成高质量标注数据。
- [ 使用介绍 ](https://linux.do/t/topic/1306118)
  
## ✨ 功能特性

### 🖼️ 图像标注

| 功能 | 描述 |
|------|------|
| **文本提示分割** | 输入中/英文描述，AI 自动识别并分割目标对象 |
| **点击分割** | 通过点击添加正/负样本点进行精确分割 |
| **框选分割** | 绘制边界框指定分割区域，支持正/负样本框 |
| **手动绘制** | 多边形工具手动绘制标注区域 |
| **批量分割** | 对多张图片进行批量自动分割 |


![屏幕截图 2025-12-13 170932](https://github.com/user-attachments/assets/a28c3a06-2c07-41ee-a605-ab35ef91a8ce)

<img width="2559" height="1285" alt="image" src="https://github.com/user-attachments/assets/2c2e1987-c7f6-45b5-9a3b-d8e5894706e3" />



### 🎯 正负样本系统

- **正样本 (绿色)**: 指示要分割的目标区域
- **负样本 (红色)**: 指示要排除的区域，用于精细化分割结果
- **智能过滤**: 使用 Mask 级别的重叠检测，精确排除不需要的分割结果

### 🤖 AI 翻译功能

- 支持中文输入，自动翻译为英文提示词
- 兼容 OpenAI API 格式（DeepSeek、通义千问、Moonshot 等）
- 可配置 API 地址、密钥和模型

### 🎬 视频标注（实验功能，默认关闭）

视频会话、文本/点击提示和传播接口可用于实验验证；帧读取、播放和导出界面仍是
占位实现，不可用于生产标注。仅在明确需要测试时启动：

```bash
SAM3_ENABLE_EXPERIMENTAL_VIDEO=1 uv run python app.py
```

### 📦 数据导出

| 格式 | 说明 |
|------|------|
| **YOLO** | 支持 YOLOv5/v8/v11 检测和分割格式 |
| **COCO** | 标准 COCO 实例分割格式 |

自动按 8:1:1 比例分割 train/val/test 数据集。
导出先在同文件系统的暂存目录完整生成，成功后再替换对应格式的旧文件；
重复导出不会混入上一次遗留的图片或标签，失败时保留上一次完整结果。
导出预览直接传输 JPEG 并在浏览器中以 Blob URL 显示，避免大图 Base64 拷贝。

## 🚀 快速开始

### 环境要求

- Python 3.11（项目通过 `.python-version` 固定；`numpy==1.26` 不支持 Python 3.13+）
- macOS Apple Silicon：使用 PyTorch MPS，已在 Apple M5 上验证文本/点/框分割
- Windows / Linux：推荐 CUDA 12.6 与 8GB+ 显存；CPU 可运行但较慢
- 权重文件 `sam3.pt` 约 3.45GB

### 安装步骤

```bash
# 1. 进入项目目录
cd Sam3-AN

# 2. 安装 uv（macOS）
brew install uv

# 3. 创建 Python 3.11 虚拟环境并安装锁定依赖
uv sync
```

> SAM3 核心代码已包含在 `SAM_src/`，无需另外安装。macOS 会安装 `eva-decord`
> （仍使用 `import decord`）；Windows/Linux 使用 `decord`。

### 下载模型权重

优先从官方 ModelScope 下载 `sam3.pt` 并放到项目根目录：
https://www.modelscope.cn/models/facebook/sam3

也可使用本项目 macOS 验证所用的公开 Hugging Face 镜像（官方
`facebook/sam3` 仓库需要登录授权）：

```bash
curl --fail --location --retry 20 --continue-at - \
  --output sam3.pt \
  https://huggingface.co/1038lab/sam3/resolve/main/sam3.pt

# 校验：文件大小 3,450,062,241 bytes
shasum -a 256 sam3.pt
# 9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
```

权重由 `.gitignore` 排除，不要提交到 Git。

### 启动服务

```bash
uv run python app.py

# 如果 MPS 出现兼容问题，可强制使用 CPU
SAM3_DEVICE=cpu uv run python app.py

# 仅验证未完成的视频会话工作流
SAM3_ENABLE_EXPERIMENTAL_VIDEO=1 uv run python app.py
```

启动后自动打开浏览器。默认使用 `http://localhost:5000`；如果 macOS AirPlay
Receiver 占用 5000，会自动依次回退到 5001/5055/8000/8080。

## 📖 使用指南

### 基本工作流程

```
创建项目 → 加载图片 → 添加类别 → 标注（自动保存）→ 导出
```

1. **创建项目**: 点击左上角项目名称，选择"项目管理"
2. **设置目录**: 配置图片目录和输出目录
3. **添加类别**: 在右侧面板添加标注类别
4. **开始标注**: 选择工具进行标注；底部状态显示自动保存进度
5. **确认导出**: 状态变为“已保存”后导出数据集
项目列表和项目详情只传输轻量图片清单；切换图片时并行加载图片与该图片的标注，
避免大项目把全部多边形一次性发送到浏览器。

### 标注工具

#### 文本提示分割 (推荐)

1. 在工具栏输入框中输入目标描述（如 "apple" 或 "苹果"）
2. 调整置信度阈值（默认 0.5）
3. 点击"分割"按钮或按 Enter

#### 点击分割

1. 选择"点击"工具
2. 选择正样本(+)或负样本(-)模式
3. 在图像上点击目标位置
4. 点击"分割"按钮执行分割

#### 框选分割

1. 选择"框选"工具
2. 选择正样本(+)或负样本(-)模式
3. 在图像上绘制边界框
4. 可添加多个正/负样本框
5. 点击"分割"按钮执行分割

#### 手动绘制

1. 选择"多边形"工具
2. 使用鼠标、触控笔或触屏点击添加多边形顶点
3. 点击工具栏的"完成多边形"按钮，或双击/按 Enter 完成
4. 完成后会自动保存；可用撤销/重做按钮修正

### 正负样本使用技巧

```
场景：图片中有多个苹果，只想标注其中一个

方法1：框选
1. 用正样本框(+)框选目标苹果
2. 用负样本框(-)框选不想要的苹果
3. 点击分割

方法2：点击
1. 用正样本点(+)点击目标苹果
2. 用负样本点(-)点击不想要的苹果
3. 点击分割
```

### 批量分割

1. 展开右侧"批量分割"面板
2. 输入提示词和目标类别
3. 设置图片范围（起始/结束索引）
4. 勾选"跳过已标注"保护已有标注
5. 点击"开始批量分割"
6. 处理中可点击“完成当前图片后停止”，已完成结果会保留

### 快捷键

| 快捷键 | 功能 |
|--------|------|
| `←` / `→` | 上一张/下一张图片 |
| `⌘/Ctrl+S` | 立即保存当前标注 |
| `⌘/Ctrl+Z` | 撤销；加 `Shift` 重做 |
| `Ctrl+Y` | 重做 |
| `Delete` | 删除选中的标注 |
| `P` / `B` / `T` / `E` / `G` | 点 / 框 / 文本 / 编辑 / 多边形工具 |
| `Enter` | 完成多边形；在文本框中执行文本分割 |
| `Backspace` | 撤销多边形最后一个顶点 |
| `Escape` | 取消临时标记或退出输入 |
| `Space` + 拖动 / 中键 / 右键 | 平移画布 |

## 🏗️ 项目结构

```
annotation_tool/
├── app.py                      # Flask 主应用
├── requirements.txt            # 依赖列表
├── sam3.pt                     # SAM3 模型权重
├── README.md                   # 项目文档
│
├── services/
│   ├── sam3_service.py         # SAM3 模型服务封装
│   └── annotation_manager.py   # 标注数据管理
│
├── exports/
│   ├── yolo_exporter.py        # YOLO 格式导出
│   └── coco_exporter.py        # COCO 格式导出
│
├── templates/
│   ├── index.html              # 图像标注页面
│   └── video.html              # 视频标注页面
│
├── static/
│   ├── css/style.css           # 赛博朋克风格样式
│   └── js/annotation.js        # 前端交互逻辑
│
├── data/                       # 项目数据存储
├── uploads/                    # 上传文件临时目录
│
├── SAM_src/                    # SAM3 源码（本地副本）

```

## ⚙️ 配置说明

### AI 翻译配置

点击工具栏的 AI 翻译配置按钮：

| 配置项 | 说明 | 示例 |
|--------|------|------|
| API 地址 | OpenAI 格式 API 地址 | `https://api.deepseek.com` |
| API 密钥 | 你的 API Key | `sk-xxx...` |
| 模型名称 | 使用的模型 | `deepseek-chat` |

支持的 API 服务：（openai格式基本都支持）
- DeepSeek: `https://api.deepseek.com`
- 通义千问: `https://dashscope.aliyuncs.com/compatible-mode`
- Moonshot: `https://api.moonshot.cn`
- OpenAI: `https://api.openai.com`

### 置信度阈值

- 范围: 0.01 - 1.0
- 默认: 0.5
- 较高值: 更精确但可能漏检
- 较低值: 更全面但可能误检

## ❓ 常见问题

### Q: 首次启动很慢？
A: 首次启动需要加载 SAM3 模型（约 3.2GB），请耐心等待。后续启动会更快。

### Q: 显存不足？
A: SAM3 需要约 6-8GB 显存。可尝试：
- 关闭其他 GPU 程序
- 使用较小的图片
- 使用 CPU 模式（较慢）

### Q: macOS 提示 MPS/CUDA 错误？
A: 默认设备优先级为 CUDA → MPS → CPU。Apple Silicon 应自动选择 MPS；可运行
`SAM3_DEVICE=cpu uv run python app.py` 强制回退 CPU。不要在 macOS 安装
`triton-windows`，项目已通过平台标记自动跳过。

### Q: macOS 的 5000 端口被占用？
A: AirPlay Receiver 经常占用 5000。程序会自动选择 5001/5055/8000/8080，
无需关闭 AirPlay；终端会打印实际访问地址。

### Q: 分割结果不准确？
A: 尝试以下方法：
- 调整置信度阈值
- 使用更精确的提示词
- 使用正负样本框/点进行精细化
- 使用英文提示词（更准确）

### Q: 中文提示词不工作？
A: 配置 AI 翻译功能，自动将中文翻译为英文。HTTPS API 会验证服务器证书；
API 密钥只保留在当前浏览器会话，关闭标签页后需重新输入，不会长期写入
`localStorage`。

### Q: 如何批量处理大量图片？
A: 使用批量分割功能，设置合适的提示词和置信度，可快速处理整个数据集。

## 🛠️ 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                    前端 (Browser)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  Canvas 渲染 │  │  工具栏交互  │  │  标注管理   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
                           │ HTTP/REST API
┌─────────────────────────────────────────────────────────┐
│                   后端 (Flask)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  路由处理    │  │  SAM3 服务   │  │  数据管理   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│                   SAM3 模型层                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  图像编码器  │  │  文本编码器  │  │  Mask 解码器 │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

标注持久化采用 schema v3：`data/projects.json` 只保存项目摘要，
`data/<id>/annotations.json` 保存图片清单，每张图片的标注独立写入
`data/<id>/image_annotations/<sha256(filename)>.json`。项目 API 返回轻量清单，
`/api/annotation/get` 按当前图片加载标注。

## 📄 许可证

MIT License

## 🙏 致谢

- [SAM3 - Segment Anything Model 3](https://github.com/facebookresearch/sam3)
- [Linuxdo](https://Linux.do/)
- [Gemini](https://gemini.google.com/)
- [ChatGPT](https://chatgpt.com/)
- [Flask](https://flask.palletsprojects.com/)
- [PyTorch](https://pytorch.org/)

---

<p align="center">
  need 小 ⭐⭐
</p>
