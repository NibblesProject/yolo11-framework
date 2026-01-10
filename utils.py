"""
工具函数模块
提供各种辅助函数
"""
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
import yaml
import json
from tqdm import tqdm


def set_seed(seed: int = 42):
    """
    设置随机种子以确保可重复性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    print(f"随机种子设置为: {seed}")


def count_files(directory: str, extensions: List[str] = None) -> int:
    """
    统计目录中的文件数量
    
    Args:
        directory: 目录路径
        extensions: 文件扩展名列表(如 ['.jpg', '.png'])
        
    Returns:
        文件数量
    """
    directory = Path(directory)
    
    if extensions is None:
        return len(list(directory.glob('*')))
    
    count = 0
    for ext in extensions:
        count += len(list(directory.glob(f'*{ext}')))
        count += len(list(directory.glob(f'*{ext.upper()}')))
    
    return count


def split_dataset(
    image_dir: str,
    label_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    划分数据集为训练集、验证集和测试集
    
    Args:
        image_dir: 图像目录
        label_dir: 标签目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    """
    # 检查比例
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1.0")
    
    # 设置随机种子
    set_seed(seed)
    
    # 获取所有图像文件
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    
    image_files = list(image_dir.glob("*.jpg")) + \
                  list(image_dir.glob("*.jpeg")) + \
                  list(image_dir.glob("*.png")) + \
                  list(image_dir.glob("*.bmp"))
    
    if len(image_files) == 0:
        raise ValueError(f"在 {image_dir} 中没有找到图像文件")
    
    # 打乱顺序
    random.shuffle(image_files)
    
    # 计算划分点
    n = len(image_files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    # 创建输出目录
    train_img_dir = output_dir / "train" / "images"
    train_label_dir = output_dir / "train" / "labels"
    val_img_dir = output_dir / "val" / "images"
    val_label_dir = output_dir / "val" / "labels"
    test_img_dir = output_dir / "test" / "images"
    test_label_dir = output_dir / "test" / "labels"
    
    for dir_path in [train_img_dir, train_label_dir, val_img_dir, val_label_dir,
                     test_img_dir, test_label_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 复制文件
    print(f"划分数据集: 总共 {n} 张图像")
    print(f"训练集: {len(train_files)} 张")
    print(f"验证集: {len(val_files)} 张")
    print(f"测试集: {len(test_files)} 张")
    
    # 复制训练集
    print("复制训练集...")
    for img_file in tqdm(train_files):
        # 复制图像
        shutil.copy2(img_file, train_img_dir / img_file.name)
        # 复制标签
        label_file = label_dir / (img_file.stem + ".txt")
        if label_file.exists():
            shutil.copy2(label_file, train_label_dir / label_file.name)
    
    # 复制验证集
    print("复制验证集...")
    for img_file in tqdm(val_files):
        # 复制图像
        shutil.copy2(img_file, val_img_dir / img_file.name)
        # 复制标签
        label_file = label_dir / (img_file.stem + ".txt")
        if label_file.exists():
            shutil.copy2(label_file, val_label_dir / label_file.name)
    
    # 复制测试集
    if len(test_files) > 0:
        print("复制测试集...")
        for img_file in tqdm(test_files):
            # 复制图像
            shutil.copy2(img_file, test_img_dir / img_file.name)
            # 复制标签
            label_file = label_dir / (img_file.stem + ".txt")
            if label_file.exists():
                shutil.copy2(label_file, test_label_dir / label_file.name)
    
    print(f"数据集划分完成! 保存到: {output_dir}")


def visualize_dataset(
    yaml_path: str,
    num_samples: int = 9,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    可视化数据集样本
    
    Args:
        yaml_path: 数据集配置文件路径
        num_samples: 可视化样本数量
        save_path: 保存路径
        show: 是否显示
    """
    # 加载配置
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data_dict = yaml.safe_load(f)
    
    train_path = Path(data_dict['train']) / 'images'
    names = data_dict['names']
    
    # 获取图像文件
    image_files = list(train_path.glob("*.jpg")) + \
                  list(train_path.glob("*.jpeg")) + \
                  list(train_path.glob("*.png"))
    
    if len(image_files) == 0:
        raise ValueError(f"在 {train_path} 中没有找到图像文件")
    
    # 随机选择样本
    random.shuffle(image_files)
    image_files = image_files[:num_samples]
    
    # 创建网格
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.ravel() if grid_size > 1 else [axes]
    
    for idx, img_file in enumerate(image_files):
        # 读取图像
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 读取标签
        label_file = img_file.parent.parent / 'labels' / (img_file.stem + '.txt')
        if label_file.exists():
            with open(label_file, 'r') as f:
                labels = f.read().strip().split('\n')
            
            h, w = img.shape[:2]
            for label in labels:
                if label:
                    parts = label.split()
                    class_id = int(parts[0])
                    x_center, y_center, bw, bh = map(float, parts[1:5])
                    
                    # 转换为像素坐标
                    x_center *= w
                    y_center *= h
                    bw *= w
                    bh *= h
                    
                    x1 = int(x_center - bw / 2)
                    y1 = int(y_center - bh / 2)
                    x2 = int(x_center + bw / 2)
                    y2 = int(y_center + bh / 2)
                    
                    # 绘制边界框
                    color = [random.randint(0, 255) for _ in range(3)]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # 绘制标签
                    label_text = names[class_id] if class_id < len(names) else str(class_id)
                    cv2.putText(img, label_text, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        axes[idx].imshow(img)
        axes[idx].axis('off')
        axes[idx].set_title(img_file.name)
    
    # 隐藏多余的子图
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def analyze_dataset(yaml_path: str):
    """
    分析数据集统计信息
    
    Args:
        yaml_path: 数据集配置文件路径
    """
    # 加载配置
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data_dict = yaml.safe_load(f)
    
    print("=" * 50)
    print("数据集分析")
    print("=" * 50)
    
    # 统计信息
    for split in ['train', 'val', 'test']:
        if split in data_dict:
            img_dir = Path(data_dict[split]) / 'images'
            label_dir = Path(data_dict[split]) / 'labels'
            
            if img_dir.exists():
                num_images = count_files(img_dir, ['.jpg', '.jpeg', '.png', '.bmp'])
                print(f"\n{split.upper()} 集:")
                print(f"  图像数量: {num_images}")
                
                if label_dir.exists():
                    num_labels = count_files(label_dir, ['.txt'])
                    print(f"  标签数量: {num_labels}")
                    
                    # 统计类别分布
                    class_counts = {}
                    label_files = list(label_dir.glob("*.txt"))
                    for label_file in label_files[:1000]:  # 采样1000个文件
                        with open(label_file, 'r') as f:
                            labels = f.read().strip().split('\n')
                        for label in labels:
                            if label:
                                class_id = int(label.split()[0])
                                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                    
                    print(f"\n  类别分布 (采样):")
                    for class_id, count in sorted(class_counts.items()):
                        class_name = data_dict['names'][class_id] if class_id < len(data_dict['names']) else str(class_id)
                        print(f"    {class_name}: {count}")
    
    print(f"\n总类别数: {data_dict['nc']}")
    print(f"类别名称: {data_dict['names']}")
    print("=" * 50)


def convert_coco_to_yolo(
    coco_json_path: str,
    output_dir: str,
    image_dir: str
):
    """
    将COCO格式转换为YOLO格式
    
    Args:
        coco_json_path: COCO标注JSON文件路径
        output_dir: 输出目录
        image_dir: 图像目录
    """
    print("COCO转YOLO格式转换...")
    # 这里可以实现COCO到YOLO的转换逻辑
    # 由于代码较长,这里只是框架
    print("功能待实现")


def get_model_size(model_path: str) -> float:
    """
    获取模型文件大小(MB)
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        文件大小(MB)
    """
    return os.path.getsize(model_path) / (1024 * 1024)


def calculate_flops(model_path: str, img_size: int = 640) -> Dict[str, float]:
    """
    计算模型的FLOPs和参数量
    
    Args:
        model_path: 模型文件路径
        img_size: 输入图像大小
        
    Returns:
        包含flops和params的字典
    """
    try:
        from model import YOLO11Model
        import torch
        
        model = YOLO11Model(weights=model_path, verbose=False)
        dummy_input = torch.randn(1, 3, img_size, img_size)
        
        # 使用torchstat或thop计算FLOPs
        try:
            from thop import profile
            flops, params = profile(model.model.model, inputs=(dummy_input,))
            return {
                'flops': flops / 1e9,  # GFLOPs
                'params': params / 1e6   # M参数
            }
        except ImportError:
            print("警告: 未安装thop库,无法计算FLOPs")
            return {
                'flops': None,
                'params': None
            }
    except Exception as e:
        print(f"计算FLOPs时出错: {e}")
        return {
            'flops': None,
            'params': None
        }


def plot_training_curves(log_dir: str, save_path: Optional[str] = None):
    """
    绘制训练曲线
    
    Args:
        log_dir: 日志目录
        save_path: 保存路径
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
        import matplotlib.pyplot as plt
        
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        
        # 获取训练指标
        tags = ea.Tags()['scalars']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        # 常见指标映射
        metric_map = {
            'train/box_loss': 'Box Loss',
            'train/cls_loss': 'Class Loss',
            'train/dfl_loss': 'DFL Loss',
            'metrics/mAP50': 'mAP50',
            'metrics/mAP50-95': 'mAP50-95',
            'metrics/precision': 'Precision',
            'metrics/recall': 'Recall'
        }
        
        idx = 0
        for tag, title in metric_map.items():
            if tag in tags and idx < 4:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                
                axes[idx].plot(steps, values)
                axes[idx].set_xlabel('Step')
                axes[idx].set_ylabel(title)
                axes[idx].set_title(title)
                axes[idx].grid(True)
                idx += 1
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"训练曲线已保存到: {save_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"绘制训练曲线时出错: {e}")


def export_results_to_csv(results: List[Dict[str, Any]], output_path: str):
    """
    将结果导出为CSV文件
    
    Args:
        results: 结果列表
        output_path: 输出CSV文件路径
    """
    try:
        import pandas as pd
        
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"结果已导出到: {output_path}")
    except ImportError:
        print("警告: 未安装pandas库,无法导出CSV")
    except Exception as e:
        print(f"导出CSV时出错: {e}")


def check_gpu():
    """检查GPU可用性"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU可用: {torch.cuda.get_device_name(0)}")
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"PyTorch版本: {torch.__version__}")
            return True
        else:
            print("GPU不可用,将使用CPU")
            return False
    except ImportError:
        print("未安装PyTorch,无法检查GPU")
        return False


def print_model_summary(model_path: str):
    """
    打印模型摘要信息
    
    Args:
        model_path: 模型文件路径
    """
    try:
        from model import YOLO11Model
        
        model = YOLO11Model(weights=model_path, verbose=False)
        info = model.get_model_info()
        
        print("=" * 50)
        print("模型摘要")
        print("=" * 50)
        print(f"模型名称: {info['model_name']}")
        print(f"参数数量: {info['parameters']:,}")
        print(f"设备: {info['device']}")
        print(f"文件大小: {get_model_size(model_path):.2f} MB")
        
        if info['classes']:
            print(f"\n类别:")
            for idx, name in info['classes'].items():
                print(f"  {idx}: {name}")
        
        print("=" * 50)
        
    except Exception as e:
        print(f"打印模型摘要时出错: {e}")


# 导入matplotlib用于可视化
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None