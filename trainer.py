"""
训练器模块
提供完整的YOLO11训练流程管理
"""
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import json
from datetime import datetime
import shutil

from config import Config, get_config
from model import YOLO11Model
from dataset import create_data_yaml, verify_dataset


class YOLO11Trainer:
    """YOLO11训练器"""
    
    def __init__(self, config: Optional[Config] = None, config_path: Optional[str] = None):
        """
        初始化训练器
        
        Args:
            config: 配置对象
            config_path: 配置文件路径
        """
        # 加载配置
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = Config.from_yaml(config_path)
        else:
            self.config = get_config()
        
        # 初始化模型
        self.model = YOLO11Model(
            model_name=self.config.model.model_name,
            pretrained=self.config.model.pretrained,
            weights=self.config.model.weights,
            device=self.config.model.device,
            verbose=self.config.model.verbose,
            local_model_dir=self.config.model.local_model_dir
        )
        
        # TensorBoard writer
        self.writer = None
        self.train_history = {}
        
        # 训练状态
        self.best_map = 0.0
        self.current_epoch = 0
        
        print("=" * 50)
        print("YOLO11 训练器初始化完成")
        print("=" * 50)
        print(f"模型: {self.config.model.model_name}")
        print(f"设备: {self.config.model.device}")
        print(f"训练轮数: {self.config.train.epochs}")
        print(f"批次大小: {self.config.data.batch_size}")
        print(f"图像大小: {self.config.data.img_size}")
        print("=" * 50)
    
    def prepare_data(self):
        """准备数据集"""
        print("\n准备数据集...")
        
        # 创建数据集YAML配置文件 - 使用绝对路径
        data_yaml_path = str(Path("data/data.yaml").resolve())
        create_data_yaml(
            yaml_path=data_yaml_path,
            train_path=str(Path(self.config.data.train_path).resolve()),
            val_path=str(Path(self.config.data.val_path).resolve()),
            nc=self.config.data.nc,
            names=self.config.data.names,
            test_path=str(Path(self.config.data.test_path).resolve()) if self.config.data.test_path else None
        )
        
        # 验证数据集
        if not verify_dataset(data_yaml_path):
            raise ValueError("数据集验证失败!")
        
        print("数据集准备完成!")
        return data_yaml_path
    
    def train(self, callbacks: Optional[Dict[str, Callable]] = None):
        """
        开始训练
        
        Args:
            callbacks: 回调函数字典 {'on_epoch_end': func, 'on_train_end': func}
        """
        print("\n" + "=" * 50)
        print("开始训练")
        print("=" * 50)
        
        # 检查多GPU配置
        if self.config.model.multi_gpu:
            print(f"\n多GPU训练模式已启用")
            print(f"使用的GPU: {self.config.model.device}")
            
            # 检查可用的GPU数量
            try:
                import torch
                if torch.cuda.is_available():
                    num_gpus = torch.cuda.device_count()
                    print(f"可用GPU数量: {num_gpus}")
                    
                    # 如果device是单个数字，自动使用所有GPU
                    if self.config.model.device.isdigit():
                        self.config.model.device = ",".join([str(i) for i in range(num_gpus)])
                        print(f"自动使用所有GPU: {self.config.model.device}")
            except ImportError:
                print("警告: 未安装PyTorch，无法检查GPU")
        
        # 准备数据
        data_yaml_path = self.prepare_data()
        
        # 创建带时间戳的实验目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{self.config.train.name}_{timestamp}"
        
        # 统一使用experiments目录管理所有实验
        exp_dir = Path("experiments") / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        weights_dir = exp_dir / "weights"
        weights_dir.mkdir(exist_ok=True)
        
        charts_dir = exp_dir / "charts"
        charts_dir.mkdir(exist_ok=True)
        
        logs_dir = exp_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Ultralytics 会直接保存到 experiments/exp_<timestamp>/
        save_dir = Path("experiments") / exp_name
        
        # 保存配置
        config_path = exp_dir / "config.yaml"
        self.config.to_yaml(str(config_path))
        
        # 初始化TensorBoard
        self.writer = SummaryWriter(str(logs_dir))
        
        print(f"\n实验目录: {exp_dir}")
        print(f"  - 权重: {weights_dir}")
        print(f"  - 图表: {charts_dir}")
        print(f"  - 日志: {logs_dir}")
        
        # 训练前回调
        if callbacks and 'on_train_start' in callbacks:
            callbacks['on_train_start'](self)
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 构建训练参数
            train_args = {
                'data': data_yaml_path,
                'epochs': self.config.train.epochs,
                'batch': self.config.data.batch_size,
                'imgsz': self.config.data.img_size,
                'patience': self.config.train.patience,
                'optimizer': self.config.train.optimizer,
                'lr0': self.config.train.lr0,
                'lrf': self.config.train.lrf,
                'momentum': self.config.train.momentum,
                'weight_decay': self.config.train.weight_decay,
                'warmup_epochs': self.config.train.warmup_epochs,
                'warmup_momentum': self.config.train.warmup_momentum,
                'warmup_bias_lr': self.config.train.warmup_bias_lr,
                'project': 'experiments',  # 统一使用 experiments 目录
                'name': exp_name,  # 使用带时间戳的实验名称
                'exist_ok': self.config.train.exist_ok,
                'save': self.config.train.save,
                'save_period': self.config.train.save_period,
                'resume': self.config.train.resume,
                'amp': self.config.train.amp,
                'close_mosaic': self.config.train.close_mosaic,
                'single_cls': self.config.train.single_cls,
                'device': self.config.model.device,
                'workers': self.config.data.workers,
                'cache': self.config.data.cache,
                'augment': self.config.data.augment,
            }
            
            # 添加数据增强参数
            augment_params = {
                'hsv_h': self.config.data.hsv_h,
                'hsv_s': self.config.data.hsv_s,
                'hsv_v': self.config.data.hsv_v,
                'degrees': self.config.data.degrees,
                'translate': self.config.data.translate,
                'scale': self.config.data.scale,
                'shear': self.config.data.shear,
                'perspective': self.config.data.perspective,
                'flipud': self.config.data.flipud,
                'fliplr': self.config.data.fliplr,
                'mosaic': self.config.data.mosaic,
                'mixup': self.config.data.mixup,
            }
            train_args.update(augment_params)
            
            # 开始训练
            results = self.model.train(**train_args)
            
            # 整理实验结果到统一目录
            self._organize_experiment_results(exp_dir, save_dir)
            
            print("\n" + "=" * 50)
            print("训练完成!")
            print("=" * 50)
            print(f"训练时间: {time.time() - start_time:.2f}秒")
            print(f"\n实验结果已保存到: {exp_dir}/")
            print(f"  - 模型权重: {weights_dir}/")
            print(f"  - 训练图表: {charts_dir}/")
            print(f"  - 日志文件: {logs_dir}/")
            print(f"  - 配置文件: {config_path}")
            
        except Exception as e:
            print(f"\n训练出错: {e}")
            raise e
        
        finally:
            # 关闭TensorBoard
            if self.writer:
                self.writer.close()
            
            # 训练结束回调
            if callbacks and 'on_train_end' in callbacks:
                callbacks['on_train_end'](self)
    
    def resume(self, checkpoint_path: str):
        """
        继续训练
        
        Args:
            checkpoint_path: 检查点路径
        """
        print(f"\n从检查点继续训练: {checkpoint_path}")
        
        # 加载检查点
        self.model.load_weights(checkpoint_path)
        
        # 设置继续训练标志
        self.config.train.resume = True
        
        # 开始训练
        self.train()
    
    def validate(self, data_yaml_path: Optional[str] = None):
        """
        验证模型
        
        Args:
            data_yaml_path: 数据集配置文件路径
        """
        print("\n" + "=" * 50)
        print("开始验证")
        print("=" * 50)
        
        if data_yaml_path is None:
            data_yaml_path = "data/data.yaml"
        
        # 验证模型
        results = self.model.validate(
            data=data_yaml_path,
            batch=self.config.data.batch_size,
            imgsz=self.config.data.img_size,
            conf=self.config.val.conf,
            iou=self.config.val.iou,
            max_det=self.config.val.max_det,
            half=self.config.val.half,
            device=self.config.model.device,
            rect=self.config.val.rect,
            plots=self.config.val.plots
        )
        
        print("\n验证结果:")
        print(f"mAP50: {results.box.map50:.4f}")
        print(f"mAP50-95: {results.box.map:.4f}")
        print(f"Precision: {results.box.mp:.4f}")
        print(f"Recall: {results.box.mr:.4f}")
        
        return results
    
    def export_model(self, format: str = 'onnx', **kwargs):
        """
        导出模型
        
        Args:
            format: 导出格式
            **kwargs: 其他参数
        """
        print(f"\n导出模型为 {format} 格式...")
        
        results = self.model.export(format=format, **kwargs)
        
        print(f"模型导出完成!")
        return results
    
    def _save_training_history(self, exp_dir: Path):
        """保存训练历史"""
        history_path = exp_dir / "training_history.json"
        
        # 这里可以从Ultralytics的训练结果中提取更多信息
        history = {
            'config': self.config.__dict__,
            'model': self.model.get_model_info(),
        }
        
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"训练历史已保存到: {history_path}")
    
    def _organize_experiment_results(self, exp_dir: Path, save_dir: Path):
        """
        整理实验结果到统一目录结构
        
        Args:
            exp_dir: 实验主目录
            save_dir: Ultralytics训练结果临时保存目录
        """
        print("\n整理实验结果...")
        
        weights_dir = exp_dir / "weights"
        charts_dir = exp_dir / "charts"
        
        # 复制模型权重文件
        ultralytics_weights = save_dir / "weights"
        if ultralytics_weights.exists():
            for pt_file in ultralytics_weights.glob("*.pt"):
                shutil.copy2(pt_file, weights_dir / pt_file.name)
                print(f"  复制权重: {pt_file.name}")
        
        # 复制图表文件（混淆矩阵、训练曲线等）
        for pattern in ["*.png", "*.jpg", "*.csv"]:
            for chart_file in save_dir.glob(pattern):
                # 跳过weights目录中的文件
                if "weights" not in str(chart_file):
                    shutil.copy2(chart_file, charts_dir / chart_file.name)
                    print(f"  复制图表: {chart_file.name}")
        
        # 复制TensorBoard日志
        # 日志已经在logs_dir中，不需要复制
        
        print(f"\n实验结果整理完成!")


def train_from_config(config_path: str):
    """
    从配置文件训练的便捷函数
    
    Args:
        config_path: 配置文件路径
    """
    # 加载配置
    config = Config.from_yaml(config_path)
    
    # 创建训练器
    trainer = YOLO11Trainer(config=config)
    
    # 开始训练
    trainer.train()


def train_simple(
    data_yaml: str,
    model_name: str = 'yolo11n',
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = '0',
    save_dir: str = 'runs/train'
):
    """
    简单训练函数
    
    Args:
        data_yaml: 数据集配置文件路径
        model_name: 模型名称
        epochs: 训练轮数
        batch_size: 批次大小
        img_size: 图像大小
        device: 设备
        save_dir: 保存目录
    """
    # 创建配置
    config = Config()
    config.model.model_name = model_name
    config.model.device = device
    config.train.epochs = epochs
    config.data.batch_size = batch_size
    config.data.img_size = img_size
    config.train.save_dir = save_dir
    
    # 创建训练器
    trainer = YOLO11Trainer(config=config)
    
    # 开始训练
    trainer.train()
