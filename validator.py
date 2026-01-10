"""
验证模块
提供模型验证和评估功能
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model import YOLO11Model
from config import Config


class YOLO11Validator:
    """YOLO11验证器"""
    
    def __init__(
        self,
        model_path: str,
        device: str = '0',
        verbose: bool = True
    ):
        """
        初始化验证器
        
        Args:
            model_path: 模型权重路径
            device: 设备
            verbose: 是否显示详细信息
        """
        self.model = YOLO11Model(
            weights=model_path,
            device=device,
            verbose=verbose
        )
        self.verbose = verbose
        
        print(f"验证器初始化完成: {model_path}")
    
    def validate(
        self,
        data_yaml: str,
        batch_size: int = 16,
        imgsz: int = 640,
        conf: float = 0.001,
        iou: float = 0.6,
        max_det: int = 300,
        half: bool = True,
        plots: bool = True,
        save_json: bool = False,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        验证模型
        
        Args:
            data_yaml: 数据集配置文件路径
            batch_size: 批次大小
            imgsz: 图像大小
            conf: 置信度阈值
            iou: IoU阈值
            max_det: 每张图最大检测数
            half: 是否使用FP16
            plots: 是否绘制曲线
            save_json: 是否保存JSON结果
            save_dir: 保存目录
            
        Returns:
            验证结果字典
        """
        print("\n" + "=" * 60)
        print("开始验证")
        print("=" * 60)
        
        # 验证模型
        results = self.model.validate(
            data=data_yaml,
            batch=batch_size,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            half=half,
            device=self.model.device,
            plots=plots
        )
        
        # 提取指标
        metrics = {
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'f1': float(results.box.f1),
            'classes': results.box.maps if hasattr(results.box, 'maps') else None
        }
        
        # 打印结果
        if self.verbose:
            print("\n验证结果:")
            print(f"  mAP50: {metrics['mAP50']:.4f}")
            print(f"  mAP50-95: {metrics['mAP50-95']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1']:.4f}")
        
        # 保存结果
        if save_json and save_dir:
            self._save_results(metrics, save_dir)
        
        return metrics
    
    def _save_results(self, metrics: Dict[str, Any], save_dir: str):
        """保存验证结果"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = save_dir / "validation_results.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n验证结果已保存到: {save_path}")
    
    def compare_models(
        self,
        model_paths: List[str],
        data_yaml: str,
        model_names: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        对比多个模型的性能
        
        Args:
            model_paths: 模型路径列表
            data_yaml: 数据集配置文件路径
            model_names: 模型名称列表(可选)
            **kwargs: 其他验证参数
            
        Returns:
            各模型的验证结果字典
        """
        print("\n" + "=" * 60)
        print("模型对比验证")
        print("=" * 60)
        
        if model_names is None:
            model_names = [Path(p).stem for p in model_paths]
        
        results = {}
        
        for model_path, model_name in zip(model_paths, model_names):
            print(f"\n验证模型: {model_name}")
            
            # 创建验证器
            validator = YOLO11Validator(
                model_path=model_path,
                device=self.model.device,
                verbose=False
            )
            
            # 验证
            metrics = validator.validate(
                data_yaml=data_yaml,
                verbose=False,
                **kwargs
            )
            
            results[model_name] = metrics
        
        # 打印对比结果
        self._print_comparison(results)
        
        return results
    
    def _print_comparison(self, results: Dict[str, Dict[str, Any]]):
        """打印对比结果"""
        print("\n" + "=" * 60)
        print("模型对比结果")
        print("=" * 60)
        
        # 表头
        header = f"{'模型名称':<20} {'mAP50':<10} {'mAP50-95':<12} {'Precision':<12} {'Recall':<12}"
        print(header)
        print("-" * 80)
        
        # 数据行
        for model_name, metrics in results.items():
            row = (
                f"{model_name:<20} "
                f"{metrics['mAP50']:<10.4f} "
                f"{metrics['mAP50-95']:<12.4f} "
                f"{metrics['precision']:<12.4f} "
                f"{metrics['recall']:<12.4f}"
            )
            print(row)
    
    def plot_comparison(
        self,
        results: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        绘制模型对比图表
        
        Args:
            results: 验证结果字典
            save_path: 保存路径
            show: 是否显示
        """
        models = list(results.keys())
        metrics = ['mAP50', 'mAP50-95', 'precision', 'recall', 'f1']
        
        # 准备数据
        data = {}
        for metric in metrics:
            data[metric] = [results[model][metric] for model in models]
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        # 子图1: 所有指标对比
        x = np.arange(len(models))
        width = 0.15
        
        for i, (metric, label) in enumerate(zip(metrics[:4], ['mAP50', 'mAP50-95', 'Precision', 'Recall'])):
            axes[0].bar(x + i * width, data[metric], width, label=label)
        
        axes[0].set_xlabel('模型')
        axes[0].set_ylabel('分数')
        axes[0].set_title('模型性能对比')
        axes[0].set_xticks(x + width * 1.5)
        axes[0].set_xticklabels(models, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 子图2: mAP对比
        axes[1].bar(models, [data['mAP50'], data['mAP50-95']], width=0.6, label=['mAP50', 'mAP50-95'])
        axes[1].set_xlabel('模型')
        axes[1].set_ylabel('mAP')
        axes[1].set_title('mAP 对比')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 子图3: Precision-Recall
        for model in models:
            axes[2].scatter(data['recall'], data['precision'], label=model, s=100)
        axes[2].set_xlabel('Recall')
        axes[2].set_ylabel('Precision')
        axes[2].set_title('Precision-Recall 散点图')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 子图4: 综合性能雷达图
        categories = ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1']
        N = len(categories)
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        ax = axes[3]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        
        for model in models:
            values = [results[model][cat.lower()] for cat in categories]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.set_title('综合性能雷达图')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n对比图表已保存到: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


def validate_simple(
    model_path: str,
    data_yaml: str,
    device: str = '0'
) -> Dict[str, Any]:
    """
    简单验证函数
    
    Args:
        model_path: 模型路径
        data_yaml: 数据集配置文件路径
        device: 设备
        
    Returns:
        验证结果
    """
    validator = YOLO11Validator(model_path=model_path, device=device)
    return validator.validate(data_yaml=data_yaml)