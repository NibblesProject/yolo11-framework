"""
模型定义模块
提供YOLO11模型的加载和管理功能
"""
from ultralytics import YOLO
from pathlib import Path
from typing import Optional, Dict, Any
import torch


class YOLO11Model:
    """YOLO11模型封装类"""
    
    # 模型大小配置
    MODEL_SIZES = {
        'yolo11n': 'yolo11n.pt',      # Nano
        'yolo11s': 'yolo11s.pt',      # Small
        'yolo11m': 'yolo11m.pt',      # Medium
        'yolo11l': 'yolo11l.pt',      # Large
        'yolo11x': 'yolo11x.pt',      # Extra Large
    }
    
    def __init__(
        self,
        model_name: str = 'yolo11n',
        pretrained: bool = True,
        weights: Optional[str] = None,
        device: str = '0',
        verbose: bool = True,
        local_model_dir: str = 'model'
    ):
        """
        初始化YOLO11模型
        
        Args:
            model_name: 模型名称 ('yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x')
            pretrained: 是否使用预训练权重
            weights: 自定义权重路径
            device: 设备 ('0', 'cpu', 'cuda:0', etc.)
            verbose: 是否显示详细信息
            local_model_dir: 本地模型目录
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self.verbose = verbose
        self.local_model_dir = local_model_dir
        
        # 构建模型路径 - 优先使用本地模型
        if weights is not None:
            model_path = weights
        else:
            # 优先从本地目录加载模型
            local_model_path = Path(local_model_dir) / self.MODEL_SIZES.get(model_name, 'yolo11n.pt')
            
            if local_model_path.exists():
                # 使用绝对路径，确保Ultralytics从本地加载而不是从网上下载
                model_path = str(local_model_path.resolve())
                if verbose:
                    print(f"使用本地模型: {model_path}")
            else:
                # 如果本地不存在，使用默认路径（Ultralytics会自动下载）
                model_path = self.MODEL_SIZES.get(model_name, 'yolo11n.pt')
                if pretrained:
                    print(f"警告: 本地模型不存在 {local_model_path}, 将使用默认路径: {model_path}")
                    print(f"将从网上下载预训练模型: {model_path}")
        
        # 加载模型
        self.model = YOLO(model_path)
        
        # 设置设备（Ultralytics会自动处理设备）
        # 不需要手动调用to(device)，在train/val/predict时指定device参数即可
        
        if verbose:
            print(f"模型加载完成: {model_name}")
            print(f"设备: {device}")
            print(f"参数量: {self._count_parameters():,}")
    
    def train(
        self,
        data: str,
        epochs: int = 100,
        batch: int = 16,
        imgsz: int = 640,
        patience: int = 50,
        optimizer: str = 'auto',
        lr0: float = 0.01,
        lrf: float = 0.01,
        momentum: float = 0.937,
        weight_decay: float = 0.0005,
        warmup_epochs: int = 3.0,
        warmup_momentum: float = 0.8,
        warmup_bias_lr: float = 0.1,
        save_dir: str = 'runs/train',
        name: str = 'exp',
        exist_ok: bool = False,
        save: bool = True,
        save_period: int = -1,
        project: str = 'yolo11',
        resume: bool = False,
        amp: bool = True,
        close_mosaic: int = 10,
        single_cls: bool = False,
        device: str = None,
        workers: int = 8,
        cache: str = 'ram',
        augment: bool = True,
        **kwargs
    ):
        """
        训练模型
        
        Args:
            data: 数据集配置文件路径
            epochs: 训练轮数
            batch: 批次大小
            imgsz: 图像大小
            patience: 早停轮数
            optimizer: 优化器
            lr0: 初始学习率
            lrf: 最终学习率
            momentum: 动量
            weight_decay: 权重衰减
            warmup_epochs: 预热轮数
            warmup_momentum: 预热动量
            warmup_bias_lr: 预热偏置学习率
            save_dir: 保存目录
            name: 实验名称
            exist_ok: 是否覆盖已有目录
            save: 是否保存检查点
            save_period: 保存频率
            project: 项目名称
            resume: 是否继续训练
            amp: 是否使用混合精度训练
            close_mosaic: 最后几轮关闭马赛克增强
            single_cls: 是否将所有类别视为同一类
            device: 设备
            workers: 数据加载线程数
            cache: 缓存方式
            augment: 是否使用数据增强
            **kwargs: 其他参数
            
        Returns:
            训练结果
        """
        if device is None:
            device = self.device
        
        # 训练参数
        train_args = {
            'data': data,
            'epochs': epochs,
            'batch': batch,
            'imgsz': imgsz,
            'patience': patience,
            'optimizer': optimizer,
            'lr0': lr0,
            'lrf': lrf,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'warmup_epochs': warmup_epochs,
            'warmup_momentum': warmup_momentum,
            'warmup_bias_lr': warmup_bias_lr,
            'project': project,
            'name': name,
            'exist_ok': exist_ok,
            'save': save,
            'save_period': save_period,
            'resume': resume,
            'amp': amp,
            'close_mosaic': close_mosaic,
            'single_cls': single_cls,
            'device': device,
            'workers': workers,
            'cache': cache,
            'augment': augment,
            'verbose': self.verbose,
        }
        
        # 添加额外参数
        train_args.update(kwargs)
        
        # 开始训练
        results = self.model.train(**train_args)
        
        return results
    
    def validate(
        self,
        data: str,
        batch: int = 16,
        imgsz: int = 640,
        conf: float = 0.001,
        iou: float = 0.6,
        max_det: int = 300,
        half: bool = True,
        device: str = None,
        rect: bool = False,
        plots: bool = True,
        **kwargs
    ):
        """
        验证模型
        
        Args:
            data: 数据集配置文件路径
            batch: 批次大小
            imgsz: 图像大小
            conf: 置信度阈值
            iou: IoU阈值
            max_det: 每张图最大检测数
            half: 是否使用FP16
            device: 设备
            rect: 是否使用矩形推理
            plots: 是否绘制曲线
            **kwargs: 其他参数
            
        Returns:
            验证结果
        """
        if device is None:
            device = self.device
        
        # 验证参数
        val_args = {
            'data': data,
            'batch': batch,
            'imgsz': imgsz,
            'conf': conf,
            'iou': iou,
            'max_det': max_det,
            'half': half,
            'device': device,
            'rect': rect,
            'plots': plots,
            'verbose': self.verbose,
        }
        
        # 添加额外参数
        val_args.update(kwargs)
        
        # 开始验证
        results = self.model.val(**val_args)
        
        return results
    
    def predict(
        self,
        source: str,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 300,
        device: str = None,
        save: bool = False,
        save_txt: bool = False,
        save_conf: bool = False,
        show: bool = False,
        stream: bool = False,
        **kwargs
    ):
        """
        推理
        
        Args:
            source: 输入源 (图像/视频/目录/URL)
            conf: 置信度阈值
            iou: IoU阈值
            max_det: 每张图最大检测数
            device: 设备
            save: 是否保存结果
            save_txt: 是否保存文本结果
            save_conf: 是否保存置信度
            show: 是否显示结果
            stream: 是否流式输出
            **kwargs: 其他参数
            
        Returns:
            推理结果
        """
        if device is None:
            device = self.device
        
        # 推理参数
        predict_args = {
            'source': source,
            'conf': conf,
            'iou': iou,
            'max_det': max_det,
            'device': device,
            'save': save,
            'save_txt': save_txt,
            'save_conf': save_conf,
            'show': show,
            'stream': stream,
            'verbose': self.verbose,
        }
        
        # 添加额外参数
        predict_args.update(kwargs)
        
        # 开始推理
        results = self.model.predict(**predict_args)
        
        return results
    
    def export(self, format: str = 'onnx', **kwargs):
        """
        导出模型
        
        Args:
            format: 导出格式 ('onnx', 'torchscript', 'engine', 'coreml', 'saved_model', 'pb', 'tflite')
            **kwargs: 其他参数
            
        Returns:
            导出结果
        """
        results = self.model.export(format=format, **kwargs)
        return results
    
    def load_weights(self, weights_path: str):
        """
        加载权重
        
        Args:
            weights_path: 权重文件路径
        """
        self.model = YOLO(weights_path)
        self.model.to(self.device)
        if self.verbose:
            print(f"权重加载完成: {weights_path}")
    
    def save_weights(self, save_path: str):
        """
        保存权重
        
        Args:
            save_path: 保存路径
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(save_path)
        if self.verbose:
            print(f"权重已保存: {save_path}")
    
    def _count_parameters(self) -> int:
        """
        计算模型参数量
        
        Returns:
            参数数量
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        info = {
            'model_name': self.model_name,
            'parameters': self._count_parameters(),
            'device': self.device,
            'classes': self.model.names if hasattr(self.model, 'names') else None,
        }
        return info


def load_model(
    model_name: str = 'yolo11n',
    pretrained: bool = True,
    weights: Optional[str] = None,
    device: str = '0'
) -> YOLO11Model:
    """
    加载YOLO11模型的便捷函数
    
    Args:
        model_name: 模型名称
        pretrained: 是否使用预训练权重
        weights: 自定义权重路径
        device: 设备
        
    Returns:
        YOLO11Model实例
    """
    return YOLO11Model(
        model_name=model_name,
        pretrained=pretrained,
        weights=weights,
        device=device
    )