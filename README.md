# YOLO11 训练框架

一个完整的 YOLO11 模型训练框架，从数据准备、模型训练到推理部署。

## 目录

- [快速开始](#快速开始)
- [安装](#安装)
- [项目结构](#项目结构)
- [训练前检查](#训练前检查)
- [开始训练](#开始训练)
- [命令行参数](#命令行参数)
- [实验目录结构](#实验目录结构)
- [监控与可视化](#监控与可视化)
- [高级用法](#高级用法)
- [故障排除](#故障排除)
- [最佳实践](#最佳实践)
- [API 文档](#api-文档)

## 安装

### 环境要求

- Python >= 3.8 （推荐 Python 3.10）
- PyTorch >= 2.0.0
- CUDA >= 11.0 (如果使用GPU)

> **我的环境配置为：**
>
> ℹ   Python版本: 3.10.19
>
> ℹ   PyTorch版本: 2.9.1+cu128
>
> ✓     CUDA可用: 是 (版本: 12.8)
>
> ℹ   GPU数量: 2
>
> ​       GPU 0: NVIDIA GeForce RTX 5060 Ti (15.93 GB)
>
> ​       GPU 1: NVIDIA GeForce RTX 5060 Ti (15.93 GB)
>
> ℹ  Ultralytics版本: 8.3.250

### 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 准备数据集

将数据集按以下结构组织：

```
data/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

标签文件格式（YOLO格式）：
```
class_id x_center y_center width height
```

创建 `data/data.yaml` 配置文件：

```yaml
# 数据集路径
train: train/images
val: valid/images

# 类别数量
nc: 6

# 类别名称
names:
  - crazing
  - inclusion
  - patches
  - pitted_surface
  - rolled-in_scale
  - scratches
```

### 2. 训练前检查

在开始训练之前，运行环境检查以确保一切正常：

```bash
python test.py
```

这将检查：
- 系统环境（Python、PyTorch、CUDA等）
- 数据集配置（data.yaml）
- 数据集完整性
- 模型加载

检查通过后会显示：
```
╔══════════════════════════════════════════════════════════════════════════════╗
║ 检查总结                                                                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
╚══════════════════════════════════════════════════════════════════════════════╝
系统环境                 ✓ 通过
数据配置                 ✓ 通过
数据集完整性               ✓ 通过
模型加载                 ✓ 通过

✓ 所有检查通过，可以开始训练！
```

### 3. 开始训练

#### 基础训练（使用默认配置）

```bash
python train.py --epochs 100 --batch-size 16 --device 0
```

#### 自定义训练参数

```bash
python train.py \
  --model yolo11s \
  --epochs 50 \
  --batch-size 32 \
  --imgsz 640 \
  --device 0 \
  --lr0 0.001 \
  --optimizer AdamW
```

#### 使用配置文件

创建 `my_config.yaml`：

```yaml
data:
  nc: 6
  names:
    - crazing
    - inclusion
    - patches
    - pitted_surface
    - rolled-in_scale
    - scratches
  batch_size: 16
  img_size: 640

model:
  model_name: yolo11n
  device: '0'
  pretrained: true

train:
  epochs: 100
  patience: 50
  optimizer: 'AdamW'
  lr0: 0.001
```

然后使用：

```bash
python train.py --config my_config.yaml
```

## 项目结构

```
yolo11/
├── config.py               # 配置管理模块
├── dataset.py              # 数据加载模块
├── model.py                # 模型定义模块
├── trainer.py              # 训练器模块
├── inference.py            # 推理模块
├── utils.py                # 工具函数模块
├── train.py                # 主训练脚本
├── test.py                 # 训练前检查脚本
├── config.yaml             # 配置文件示例
├── requirements.txt        # 依赖列表
├── data/                   # 数据集目录
│   ├── data.yaml          # 数据集配置
│   ├── train/             # 训练数据
│   └── valid/             # 验证数据
└── experiments/            # 实验结果目录（统一管理）
    └── exp_20260110_122657/
        ├── weights/        # 模型权重
        ├── charts/         # 训练图表
        ├── logs/           # TensorBoard日志
        ├── config.yaml     # 训练配置
        └── training_history.json
```

## 训练前检查

运行 `python test.py` 会执行以下检查：

### 1. 系统环境检查
- Python 版本
- PyTorch 版本
- CUDA 可用性和版本
- GPU 信息
- Ultralytics 版本

### 2. 数据集配置检查
- 类别名称和数量
- 训练路径和图像数量
- 验证路径和图像数量

### 3. 数据集完整性检查
- 图像和标签文件对应
- 标注文件格式验证

### 4. 模型加载检查
- 本地模型文件检查
- 模型加载测试

### 跳过检查

如果确定环境正确，可以跳过检查：

```bash
python train.py --no-check
```

## 开始训练

### 基础训练

```bash
python train.py
```

### 从检查点恢复训练

```bash
python train.py \
  --mode resume \
  --resume-path experiments/exp_20260110_122657/weights/last.pt
```

### 模型验证

```bash
python train.py \
  --mode validate \
  --data data/data.yaml \
  --weights experiments/exp_20260110_122657/weights/best.pt
```

### 模型导出

```bash
python train.py \
  --mode export \
  --weights experiments/exp_20260110_122657/weights/best.pt \
  --export-format onnx
```

支持的导出格式：
- `onnx` - ONNX 格式
- `torchscript` - TorchScript 格式
- `engine` - TensorRT 引擎
- `coreml` - CoreML 格式
- `saved_model` - TensorFlow SavedModel
- `pb` - TensorFlow PB 格式
- `tflite` - TFLite 格式

## 命令行参数

### 基本参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | None | 配置文件路径（YAML格式）|
| `--mode` | train | 运行模式：train/resume/validate/export |
| `--no-check` | False | 跳过训练前检查 |
| `--monitor` | False | 训练后启动TensorBoard和GPU监控 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 100 | 训练轮数 |
| `--batch-size` | 16 | 批次大小 |
| `--imgsz` | 640 | 图像大小 |
| `--patience` | 50 | 早停轮数 |
| `--optimizer` | auto | 优化器：auto/SGD/Adam/AdamW |
| `--lr0` | 0.01 | 初始学习率 |
| `--lrf` | 0.01 | 最终学习率 |

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | yolo11n | 模型大小：yolo11n/s/m/l/x |
| `--weights` | None | 预训练权重路径 |
| `--device` | 0 | 设备：0, 1, cpu, cuda:0 |
| `--no-pretrained` | False | 不使用预训练权重 |

### 数据参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | data/data.yaml | 数据集配置文件路径 |
| `--train-path` | None | 训练数据路径 |
| `--val-path` | None | 验证数据路径 |
| `--nc` | None | 类别数量 |

### 其他参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--seed` | 42 | 随机种子 |
| `--workers` | 8 | 数据加载线程数 |
| `--amp` | True | 使用混合精度训练 |
| `--project` | yolo11 | 项目名称 |
| `--name` | exp | 实验名称 |
| `--save-dir` | runs/train | 保存目录 |
| `--exist-ok` | False | 覆盖已有目录 |

## 实验目录结构

项目使用 `experiments/` 目录统一管理所有训练实验。

```
experiments/
└── exp_20260110_122657/        # 单次实验（按时间戳命名）
    ├── weights/                # 模型权重文件
    │   ├── best.pt            # 最佳模型
    │   └── last.pt            # 最后一个epoch的模型
    ├── charts/                 # 训练图表
    │   ├── results.png         # 训练曲线图
    │   ├── confusion_matrix.png
    │   ├── confusion_matrix_normalized.png
    │   ├── BoxF1_curve.png
    │   ├── BoxP_curve.png
    │   ├── BoxPR_curve.png
    │   ├── BoxR_curve.png
    │   ├── *.csv              # 详细数据
    │   └── *.jpg              # 批次可视化图片
    ├── logs/                   # TensorBoard日志
    │   └── events.out.tfevents.*
    ├── config.yaml             # 训练配置文件
    └── training_history.json   # 训练历史记录
```

### 目录说明

#### weights/
- **best.pt**: 验证集上表现最好的模型权重
- **last.pt**: 训练结束时最后一个epoch的模型权重
- 用于模型部署和推理

#### charts/
- **results.png**: 训练过程中的指标曲线（mAP、Precision、Recall等）
- **confusion_matrix.png**: 混淆矩阵图
- **confusion_matrix_normalized.png**: 归一化的混淆矩阵
- **Box*_curve.png**: F1、Precision、Recall曲线
- **train_batch*.jpg**: 训练批次可视化
- **val_batch*.jpg**: 验证批次可视化
- **results.csv**: 详细的训练指标数据

#### logs/
- TensorBoard事件日志文件
- 使用 `tensorboard --logdir experiments/exp_20260110_122657/logs` 查看

#### config.yaml
- 保存本次训练的所有配置参数
- 用于复现实验结果

#### training_history.json
- 训练历史的JSON格式记录
- 包含配置信息和模型信息

### 查看实验结果

```bash
# 列出所有实验
ls -la experiments/

# 查看某个实验的权重
ls experiments/exp_20260110_122657/weights/

# 查看训练图表
ls experiments/exp_20260110_122657/charts/

# 查看训练配置
cat experiments/exp_20260110_122657/config.yaml
```

## 监控与可视化

### TensorBoard

#### 启动TensorBoard

```bash
# 查看特定实验
tensorboard --logdir experiments/exp_20260110_134646/logs

# 查看所有实验
tensorboard --logdir experiments/*/logs

# 指定端口
tensorboard --logdir experiments/*/logs --port 6006
```

#### 访问TensorBoard

在浏览器中打开：http://localhost:6006

TensorBoard 显示：
- 损失曲线（train/val_loss）
- mAP曲线（mAP50, mAP50-95）
- Precision/Recall曲线
- 学习率变化
- 训练/验证指标

#### 自动启动TensorBoard

训练完成后自动启动TensorBoard：

```bash
python train.py --epochs 100 --monitor
```

### GPU监控

#### 实时查看GPU状态

```bash
watch -n 1 nvidia-smi
```

#### 自动启动GPU监控

使用 `--monitor` 参数会同时启动TensorBoard和nvidia-smi监控：

```bash
python train.py --monitor
```

监控显示：
- GPU使用率
- 显存占用
- 温度
- 功耗

### 美化的命令行输出

训练脚本使用彩色ASCII边框和图标显示信息：

- **ℹ 蓝色** - 信息提示
- **✓ 绿色** - 成功消息
- **⚠ 黄色** - 警告消息
- **✗ 红色** - 错误消息

输出示例：

```
═════════════════════════════════════════════════════════════════════════════════════════
                                YOLO11 训练框架
═════════════════════════════════════════════════════════════════════════════════════════

────────────────────────────────────────────────────────────────────────────────────────────────
运行训练前检查
────────────────────────────────────────────────────────────────────────────────────────────────

╔══════════════════════════════════════════════════════════════════════════════╗
║ 系统环境检查                                                                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
╚══════════════════════════════════════════════════════════════════════════════╝
ℹ Python版本: 3.10.19
ℹ PyTorch版本: 2.9.1+cu128
✓ CUDA可用: 是 (版本: 12.8)
```

注意：所有输出都支持中英文混合，边框完美对齐！

## 高级用法

### 自定义训练

```python
from config import Config
from trainer import YOLO11Trainer

# 创建自定义配置
config = Config()
config.model.model_name = "yolo11m"
config.train.epochs = 200
config.data.batch_size = 32
config.data.img_size = 800

# 创建训练器并训练
trainer = YOLO11Trainer(config=config)
trainer.train()
```

### 多GPU训练

框架支持多GPU并行训练。

#### 命令行方式

```bash
python train.py --device 0,1,2,3
```

#### 配置文件方式

```yaml
model:
  device: "0,1,2,3"
  multi_gpu: true
```

#### 多GPU训练建议

1. **batch size设置**：总batch size = 单GPU batch size × GPU数量
   - 例如：4个GPU，每个GPU batch_size=16，则总batch_size=64
   
2. **学习率调整**：使用多GPU时建议适当调整学习率
   - 可以设置 `lr0` 为原来的N倍（N为GPU数量）

3. **检查GPU状态**：
   ```bash
   nvidia-smi  # 查看GPU使用情况
   ```

### 数据集工具

#### 数据集划分

```python
from utils import split_dataset

split_dataset(
    image_dir="original/images",
    label_dir="original/labels",
    output_dir="data",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
)
```

#### 数据集分析

```python
from utils import analyze_dataset

analyze_dataset("data/data.yaml")
```

#### 数据集可视化

```python
from utils import visualize_dataset

visualize_dataset(
    yaml_path="data/data.yaml",
    num_samples=9,
    save_path="dataset_vis.png",
    show=True
)
```

### 模型推理

#### 推理单张图像

```python
from inference import YOLO11Inference

# 初始化推理器
inferencer = YOLO11Inference(
    model_path="experiments/exp_20260110_122657/weights/best.pt",
    device="0",
    conf=0.25
)

# 推理
result = inferencer.predict_image("path/to/image.jpg")
print(result)
```

#### 批量推理

```python
from inference import YOLO11Inference

inferencer = YOLO11Inference("experiments/exp_20260110_122657/weights/best.pt")
results = inferencer.predict_batch([
    "image1.jpg",
    "image2.jpg",
    "image3.jpg"
])
```

#### 视频推理

```python
from inference import YOLO11Inference

inferencer = YOLO11Inference("experiments/exp_20260110_122657/weights/best.pt")
inferencer.predict_video(
    video_path="input.mp4",
    output_path="output.mp4",
    save_video=True
)
```

### 混合精度训练

默认启用AMP，可以提高训练速度并减少显存使用。

要禁用：

```bash
python train.py --no-amp
```

### 模型大小对比

| 模型 | 参数量 | 速度 (ms) | mAPval | 适用场景 |
|------|--------|-----------|--------|----------|
| YOLO11n | 2.6M | 1.5 | 53.5 | 边缘设备、移动端 |
| YOLO11s | 9.4M | 2.5 | 59.5 | 实时应用 |
| YOLO11m | 20.1M | 5.0 | 64.5 | 平衡性能和速度 |
| YOLO11l | 25.3M | 6.5 | 67.5 | 高精度需求 |
| YOLO11x | 56.9M | 12.0 | 71.5 | 最高精度 |

## 故障排除

### 训练前检查失败

如果 `python test.py` 检查失败：

1. **数据集配置错误**
   - 检查 `data/data.yaml` 是否存在
   - 验证路径是否正确
   - 确认图像文件存在

2. **模型加载失败**
   - 检查网络连接（首次需要下载模型）
   - 验证本地模型路径

3. **CUDA不可用**
   - 检查GPU驱动是否安装
   - 验证CUDA与PyTorch版本兼容性

### 内存不足 (CUDA OOM)

如果遇到CUDA OOM错误：

1. 减小批次大小：
```bash
python train.py --batch-size 8
```

2. 减小图像尺寸：
```bash
python train.py --imgsz 512
```

3. 使用更小的模型：
```bash
python train.py --model yolo11n
```

4. 启用混合精度训练：
```bash
python train.py --amp
```

### 训练不收敛

1. 调整学习率
```bash
python train.py --lr0 0.001 --lrf 0.01
```

2. 检查数据标注质量

3. 增加训练轮数
```bash
python train.py --epochs 200
```

## 最佳实践

### 1. 训练流程

1. **训练前检查**
   - 始终先运行 `python test.py`
   - 确保环境配置正确

2. **开始训练**
   - 从小模型开始（yolo11n）
   - 使用合适的batch size
   - 启用混合精度训练

3. **监控训练**
   - 使用 `--monitor` 参数启动TensorBoard
   - 定期查看GPU使用情况
   - 观察损失曲线判断训练状态

4. **结果分析**
   - 查看混淆矩阵了解类别混淆情况
   - 分析训练曲线判断是否过拟合
   - 对比不同实验的结果

### 2. 实验管理

1. **使用有意义的实验名称**
   ```bash
   python train.py --name baseline_test --epochs 50
   ```

2. **保存配置文件**
   - 每个实验自动保存config.yaml
   - 用于复现实验结果

3. **定期备份**
   - 定期备份best.pt模型
   - 保存重要实验的结果

4. **实验记录**
   - 记录每次实验的目的和参数
   - 记录实验结果和观察

### 3. 模型选择

1. **根据需求选择模型**
   - 边缘设备：yolo11n
   - 实时应用：yolo11s
   - 平衡场景：yolo11m
   - 高精度：yolo11l/x

2. **先小后大**
   - 先用小模型快速验证
   - 再用大模型提升精度

### 4. 数据增强

合理使用数据增强可以提升模型泛化能力：

```yaml
data_aug:
  hsv_h: 0.015  # 色调增强
  hsv_s: 0.7     # 饱和度增强
  hsv_v: 0.4     # 明度增强
  degrees: 0.0   # 旋转
  translate: 0.1  # 平移
  scale: 0.5     # 缩放
  flipud: 0.0    # 上下翻转
  fliplr: 0.5    # 左右翻转
  mosaic: 1.0    # 马赛克增强
  mixup: 0.0     # Mixup增强
```

## API 文档

### Config 类

配置管理类，用于管理训练的所有参数。

```python
from config import Config, get_config

# 加载默认配置
config = get_config()

# 从 YAML 文件加载
config = Config.from_yaml("config.yaml")

# 保存配置到文件
config.to_yaml("my_config.yaml")
```

### YOLO11Trainer 类

训练器类，封装了完整的训练流程。

```python
from trainer import YOLO11Trainer
from config import Config

trainer = YOLO11Trainer(config=config)
trainer.train()
```

### YOLO11Inference 类

推理器类，提供模型推理功能。

```python
from inference import YOLO11Inference

inferencer = YOLO11Inference(model_path="model.pt")
result = inferencer.predict_image("image.jpg")
```

## 更新日志

### 最新更新

- ✅ 统一使用 `data/data.yaml` 管理数据集配置
- ✅ 统一使用 `experiments/` 目录管理实验结果
- ✅ 整合测试文件为 `test.py`，支持完整的环境检查
- ✅ 美化命令行输出，支持中英文混合完美对齐
- ✅ 训练前自动检查，确保环境正确
- ✅ 支持TensorBoard和nvidia-smi自动监控
- ✅ 完整的使用文档和示例

