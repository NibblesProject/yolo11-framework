"""
配置管理模块
用于管理YOLO11训练的各种配置参数
"""
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DataConfig:
    """数据配置"""
    data_yaml_path: str = "data/data.yaml"  # 数据集配置文件路径（YAML格式）
    train_path: str = "data/train"  # 训练数据路径
    val_path: str = "data/val"      # 验证数据路径
    test_path: Optional[str] = None # 测试数据路径
    nc: int = 80                    # 类别数量
    names: Optional[List[str]] = None  # 类别名称列表，如果为None则从data.yaml读取
    classes_file: str = "dataset_classes.txt"  # 类别名称文件路径（已弃用，保留兼容性）
    img_size: int = 640             # 输入图像大小
    batch_size: int = 16            # 批次大小
    workers: int = 8                # 数据加载线程数
    cache: str = "ram"              # 数据缓存方式: 'ram', 'disk', or None
    augment: bool = True            # 是否使用数据增强
    hsv_h: float = 0.015            # HSV色调增强
    hsv_s: float = 0.7              # HSV饱和度增强
    hsv_v: float = 0.4              # HSV明度增强
    degrees: float = 0.0            # 旋转角度范围
    translate: float = 0.1          # 平移范围
    scale: float = 0.5              # 缩放范围
    shear: float = 0.0              # 剪切角度范围
    perspective: float = 0.0        # 透视变换范围
    flipud: float = 0.0             # 上下翻转概率
    fliplr: float = 0.5             # 左右翻转概率
    mosaic: float = 1.0             # 马赛克增强概率
    mixup: float = 0.0              # mixup增强概率


@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = "yolo11n"     # 模型大小: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
    pretrained: bool = True         # 是否使用预训练权重
    weights: Optional[str] = None   # 自定义权重路径
    local_model_dir: str = "model"  # 本地模型目录，优先从此目录加载模型
    device: str = '0'               # 设备: '0', '1', 'cpu', 'cuda:0', '0,1,2,3' (多GPU)
    verbose: bool = True            # 是否显示详细信息
    multi_gpu: bool = False        # 是否使用多GPU训练
    ddp: bool = False             # 是否使用分布式数据并行(DDP)


@dataclass
class TrainConfig:
    """训练配置"""
    epochs: int = 100               # 训练轮数
    patience: int = 50              # 早停轮数
    optimizer: str = "auto"         # 优化器: 'auto', 'SGD', 'Adam', 'AdamW'
    lr0: float = 0.01               # 初始学习率
    lrf: float = 0.01               # 最终学习率
    momentum: float = 0.937         # SGD动量
    weight_decay: float = 0.0005    # 权重衰减
    warmup_epochs: int = 3.0        # 预热轮数
    warmup_momentum: float = 0.8    # 预热动量
    warmup_bias_lr: float = 0.1     # 预热偏置学习率
    save_dir: str = "runs/train"    # 保存目录
    name: str = "exp"               # 实验名称
    exist_ok: bool = False          # 是否覆盖已有目录
    save: bool = True               # 是否保存检查点
    save_period: int = -1           # 保存频率(轮数), -1表示只在最后一轮保存
    project: str = "yolo11"         # 项目名称
    resume: bool = False            # 是否继续训练
    amp: bool = True                # 是否使用混合精度训练
    close_mosaic: int = 10         # 最后几轮关闭马赛克增强
    single_cls: bool = False        # 是否将所有类别视为同一类


@dataclass
class ValConfig:
    """验证配置"""
    val: bool = True                # 是否验证
    rect: bool = False              # 是否使用矩形推理
    conf: float = 0.001             # 置信度阈值
    iou: float = 0.6                # IoU阈值
    max_det: int = 300              # 每张图最大检测数
    half: bool = True               # 是否使用FP16
    dnn: bool = False               # 是否使用OpenCV DNN
    plots: bool = True              # 是否绘制曲线


@dataclass
class Config:
    """总配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    val: ValConfig = field(default_factory=ValConfig)
    
    def _load_classes_from_yaml(self, yaml_path: Optional[str] = None) -> List[str]:
        """
        从YAML文件读取数据集配置（包括类别名称、路径、数量等）
        
        Args:
            yaml_path: YAML配置文件路径，如果为None则使用self.data.data_yaml_path
            
        Returns:
            类别名称列表
        """
        if yaml_path is None:
            yaml_path = self.data.data_yaml_path
        
        yaml_file = Path(yaml_path)
        if yaml_file.exists():
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)
            
            # 读取类别名称
            if 'names' in data_config:
                names = data_config['names']
                if names:
                    self.data.names = names
                    print(f"从 {yaml_path} 读取了 {len(names)} 个类别: {names}")
            
            # 读取类别数量
            if 'nc' in data_config:
                self.data.nc = data_config['nc']
                print(f"类别数量: {self.data.nc}")
            
            # 读取训练路径
            if 'train' in data_config:
                self.data.train_path = data_config['train']
                print(f"训练路径: {self.data.train_path}")
            
            # 读取验证路径
            if 'val' in data_config:
                self.data.val_path = data_config['val']
                print(f"验证路径: {self.data.val_path}")
            
            return self.data.names if self.data.names else []
        
        # 如果YAML文件不存在，尝试从旧的txt文件读取（向后兼容）
        print(f"警告: 未找到数据集配置文件 {yaml_path}")
        return self._load_classes_from_file()
    
    def _load_classes_from_file(self, classes_file: Optional[str] = None) -> List[str]:
        """
        从文件读取类别名称（已弃用，保留向后兼容性）
        
        Args:
            classes_file: 类别文件路径，如果为None则使用self.data.classes_file
            
        Returns:
            类别名称列表
        """
        if classes_file is None:
            classes_file = self.data.classes_file
        
        file_path = Path(classes_file)
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                names = [line.strip() for line in f if line.strip()]
            if names:
                self.data.nc = len(names)
                print(f"从 {classes_file} 读取了 {len(names)} 个类别（已弃用，建议使用 data.yaml）")
                return names
        
        # 如果文件不存在，返回默认值
        print(f"警告: 未找到类别文件 {classes_file}，使用默认类别名称")
        return [f"class_{i}" for i in range(self.data.nc)]
    
    def __post_init__(self):
        """初始化后处理"""
        if self.data.names is None:
            # 优先从YAML文件读取，不存在则尝试从txt文件读取（向后兼容）
            self.data.names = self._load_classes_from_yaml()
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """从YAML文件加载配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        config = cls()
        
        # 更新data配置
        if 'data' in config_dict:
            data_dict = config_dict['data']
            for key, value in data_dict.items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)
        
        # 如果names为None或为空列表，从YAML文件读取
        if config.data.names is None or (isinstance(config.data.names, list) and len(config.data.names) == 0):
            config.data.names = config._load_classes_from_yaml()
        
        # 更新model配置
        if 'model' in config_dict:
            model_dict = config_dict['model']
            for key, value in model_dict.items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
        
        # 更新train配置
        if 'train' in config_dict:
            train_dict = config_dict['train']
            for key, value in train_dict.items():
                if hasattr(config.train, key):
                    setattr(config.train, key, value)
        
        # 更新val配置
        if 'val' in config_dict:
            val_dict = config_dict['val']
            for key, value in val_dict.items():
                if hasattr(config.val, key):
                    setattr(config.val, key, value)
        
        return config
    
    def to_yaml(self, yaml_path: str):
        """保存配置到YAML文件"""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'train': self.train.__dict__,
            'val': self.val.__dict__
        }
        
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)


def get_config(config_path: Optional[str] = None) -> Config:
    """获取配置对象"""
    if config_path and Path(config_path).exists():
        return Config.from_yaml(config_path)
    return Config()