# YOLO11 配置修改总结

## 修改内容

### 1. 模型路径修改 - 优先使用本地模型

#### 修改文件：`model.py`
- **修改位置**：`YOLO11Model.__init__` 方法
- **修改内容**：
  - 添加了 `local_model_dir` 参数，默认值为 `"model"`
  - 优先从 `local_model_dir` 目录加载模型文件
  - 如果本地模型不存在，才会使用默认路径（Ultralytics会自动下载）
  - 添加了详细的日志输出，提示使用的是本地模型还是在线下载

#### 修改文件：`config.py`
- **修改位置**：`ModelConfig` 数据类
- **修改内容**：
  - 添加了 `local_model_dir: str = "model"` 字段
  - 支持从配置文件读取本地模型目录路径

#### 修改文件：`config.yaml`
- **修改位置**：`model` 配置节
- **修改内容**：
  - 添加了 `local_model_dir: "model"` 配置项
  - 用户可以通过修改配置文件来指定本地模型目录

#### 修改文件：`trainer.py`
- **修改位置**：`YOLO11Trainer.__init__` 方法
- **修改内容**：
  - 在初始化模型时传递 `local_model_dir` 参数

### 2. 日志输出路径修改 - 使用logs文件夹并包含时间戳

#### 修改文件：`trainer.py`
- **修改位置**：`YOLO11Trainer.train` 方法
- **修改内容**：
  - 导入了 `datetime` 模块
  - 创建带时间戳的日志目录：`logs/{name}_{timestamp}`
  - 时间戳格式：`%Y%m%d_%H%M%S`（例如：20260109_225612）
  - 同时也修改了模型保存目录，使用相同的时间戳命名
  - 添加了日志输出信息，显示日志和模型的保存路径

#### 修改文件：目录结构
- **新增目录**：`logs/`
  - 所有训练的TensorBoard日志将保存在此目录下
  - 每次训练都会创建一个带时间戳的子目录

### 3. 训练结果自动整理 - 使用results和results_map文件夹

#### 修改文件：`trainer.py`
- **修改位置**：`YOLO11Trainer.train` 和 `_copy_results_to_directories` 方法
- **修改内容**：
  - 导入了 `shutil` 模块用于文件复制
  - 训练完成后自动调用 `_copy_results_to_directories` 方法
  - 创建带时间戳的 `results/` 和 `results_map/` 目录
  - `results/{name}_{timestamp}/` 目录保存：
    - 模型权重文件（`weights/*.pt`）
    - 配置文件（`config.yaml`）
    - 训练历史（`training_history.json`）
  - `results_map/{name}_{timestamp}/` 目录保存：
    - 训练图表（`*.png`, `*.jpg`）
    - 训练数据文件（`*.csv`, `*.json`）
  - 显示复制进度和最终保存位置

#### 修改文件：目录结构
- **新增目录**：`results/` 和 `results_map/`
  - `results/`：保存训练好的模型和相关配置
  - `results_map/`：保存训练过程中的图表和可视化结果
  - 每次训练都会创建带时间戳的子目录

## 使用说明

### 1. 使用本地模型

将预训练模型文件放置在 `model/` 目录下：
```
model/
├── yolo11n.pt
├── yolo11s.pt
├── yolo11m.pt
├── yolo11l.pt
└── yolo11x.pt
```

在 `config.yaml` 中配置：
```yaml
model:
  model_name: "yolo11n"
  pretrained: true
  weights: null
  local_model_dir: "model"  # 本地模型目录
```

训练时会自动：
1. 首先检查 `model/yolo11n.pt` 是否存在
2. 如果存在，使用本地模型
3. 如果不存在，显示警告并使用默认路径（可能触发下载）

### 2. 自定义本地模型目录

如果想使用其他目录存放模型，可以修改 `config.yaml`：
```yaml
model:
  local_model_dir: "/path/to/your/models"
```

### 3. 日志和模型保存位置

每次训练后：
- **日志**：`logs/exp_20260109_225612/`
- **模型**：`runs/train/yolo11/exp_20260109_225612/`

其中 `exp` 是 `config.yaml` 中 `train.name` 的值，`20260109_225612` 是训练开始时的时间戳。

### 4. 查看训练日志

使用TensorBoard查看训练进度：
```bash
tensorboard --logdir=logs
```

## 优势

1. **离线训练**：不需要每次都从网络下载模型，适合离线环境
2. **版本控制**：可以管理不同版本的预训练模型
3. **训练历史追踪**：每次训练都有带时间戳的独立日志目录
4. **避免冲突**：不会覆盖之前的训练日志
5. **易于归档**：可以轻松查找特定时间的训练记录

## 配置示例

完整的使用示例：

```python
from trainer import YOLO11Trainer
from config import Config

# 方式1：从配置文件加载
trainer = YOLO11Trainer(config_path="config.yaml")
trainer.train()

# 方式2：程序化配置
config = Config()
config.model.local_model_dir = "model"  # 本地模型目录
config.model.model_name = "yolo11n"
config.train.name = "my_experiment"    # 实验名称（会出现在日志路径中）

trainer = YOLO11Trainer(config=config)
trainer.train()
```

## 注意事项

1. 确保 `model/` 目录下有对应的 `.pt` 文件
2. 确保有写入 `logs/` 目录的权限
3. 时间戳使用训练开始的时间，确保每次训练都有唯一标识
4. 如果本地模型不存在，程序会给出警告并尝试使用默认路径

### 5. 训练结果自动整理

训练完成后，程序会自动将结果整理到：
- `results/{name}_{timestamp}/`：模型权重、配置文件和训练历史
- `results_map/{name}_{timestamp}/`：训练图表和可视化结果

这样可以方便地管理和查看不同实验的训练结果。
