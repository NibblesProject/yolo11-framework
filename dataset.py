"""
数据加载模块
提供YOLO格式的数据加载和处理功能
"""
import yaml
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image


class YOLODataset:
    """YOLO格式数据集"""
    
    def __init__(
        self,
        img_dir: str,
        label_dir: str,
        img_size: int = 640,
        augment: bool = True,
        cache: str = None,
        **kwargs
    ):
        """
        初始化数据集
        
        Args:
            img_dir: 图像目录
            label_dir: 标签目录
            img_size: 目标图像大小
            augment: 是否使用数据增强
            cache: 缓存方式: 'ram', 'disk', or None
            **kwargs: 其他增强参数
        """
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.augment = augment
        self.cache = cache
        self.augment_params = kwargs
        
        # 检查目录是否存在
        if not self.img_dir.exists():
            raise FileNotFoundError(f"图像目录不存在: {self.img_dir}")
        
        # 获取所有图像文件
        self.img_files = list(self.img_dir.glob("*.jpg")) + \
                         list(self.img_dir.glob("*.jpeg")) + \
                         list(self.img_dir.glob("*.png")) + \
                         list(self.img_dir.glob("*.bmp"))
        
        if len(self.img_files) == 0:
            raise ValueError(f"在 {self.img_dir} 中没有找到图像文件")
        
        # 缓存
        self.imgs = [None] * len(self.img_files) if cache == 'ram' else None
        self.ims = [None] * len(self.img_files) if cache == 'ram' else None
        
        print(f"加载了 {len(self.img_files)} 张图像")
    
    def __len__(self) -> int:
        return len(self.img_files)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        获取一个样本
        
        Returns:
            img: 图像 (H, W, C)
            labels: 标签 (n, 6) [class, x, y, w, h, rotated]
            img_path: 图像路径
        """
        # 加载图像
        img, (h0, w0), (h, w) = self.load_image(idx)
        
        # 加载标签
        labels = self.load_labels(idx, h, w)
        
        # 数据增强
        if self.augment:
            img, labels = self.augment_image(img, labels)
        
        return img, labels, str(self.img_files[idx])
    
    def load_image(self, idx: int) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """
        加载图像并调整大小
        
        Returns:
            img: 调整大小后的图像
            (h0, w0): 原始图像大小
            (h, w): 调整后的大小
        """
        img_path = self.img_files[idx]
        
        if self.cache == 'ram' and self.imgs[idx] is not None:
            # 从缓存加载
            img, labels, shapes = self.imgs[idx]
            return img, (shapes[0], shapes[1]), shapes[:2]
        
        # 读取图像
        im = cv2.imread(str(img_path))
        if im is None:
            raise ValueError(f"无法读取图像: {img_path}")
        
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h0, w0 = im.shape[:2]  # 原始大小
        
        # 计算调整大小
        r = self.img_size / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=interp)
        
        h, w = im.shape[:2]
        
        # 填充到正方形
        if (h != self.img_size) or (w != self.img_size):
            dw, dh = self.img_size - w, self.img_size - h
            dw /= 2
            dh /= 2
            im = cv2.copyMakeBorder(
                im, int(dh + 0.5), int(dh + 0.5), int(dw + 0.5), int(dw + 0.5),
                cv2.BORDER_CONSTANT, value=(114, 114, 114)
            )
        
        # 归一化
        im = im.astype(np.float32) / 255.0
        im = im.transpose(2, 0, 1)  # HWC -> CHW
        
        # 缓存
        if self.cache == 'ram':
            self.imgs[idx] = (im, None, (h, w, h0, w0))
        
        return im, (h0, w0), (self.img_size, self.img_size)
    
    def load_labels(self, idx: int, h: int, w: int) -> np.ndarray:
        """
        加载标签
        
        Args:
            idx: 样本索引
            h: 图像高度
            w: 图像宽度
            
        Returns:
            labels: 标签数组 (n, 5) [class, x_center, y_center, width, height]
        """
        label_path = self.label_dir / (self.img_files[idx].stem + ".txt")
        
        if not label_path.exists():
            return np.zeros((0, 5), dtype=np.float32)
        
        # 读取标签
        with open(label_path, 'r') as f:
            labels = f.read().strip().split('\n')
        
        if len(labels) == 0 or labels[0] == '':
            return np.zeros((0, 5), dtype=np.float32)
        
        # 解析标签
        labels = np.array([x.split() for x in labels], dtype=np.float32)
        
        # 确保标签格式正确
        if labels.shape[1] != 5:
            raise ValueError(f"标签格式错误: {label_path}")
        
        return labels
    
    def augment_image(self, img: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        数据增强
        
        Args:
            img: 图像 (C, H, W)
            labels: 标签 (n, 5)
            
        Returns:
            img: 增强后的图像
            labels: 增强后的标签
        """
        # 这里可以添加各种增强方法
        # 由于Ultralytics库已经内置了强大的数据增强，这里只是示例
        
        # HSV增强
        if self.augment_params.get('augment', True):
            img = self._hsv_augment(img)
        
        # 翻转
        if self.augment_params.get('fliplr', 0.5) > 0:
            if np.random.random() < self.augment_params['fliplr']:
                img = np.fliplr(img)
                if len(labels):
                    labels[:, 1] = 1 - labels[:, 1]  # x_center
        
        return img, labels
    
    def _hsv_augment(self, img: np.ndarray) -> np.ndarray:
        """HSV颜色增强"""
        img_hsv = cv2.cvtColor(img.transpose(1, 2, 0).astype(np.float32), cv2.COLOR_RGB2HSV)
        
        h = self.augment_params.get('hsv_h', 0.015)
        s = self.augment_params.get('hsv_s', 0.7)
        v = self.augment_params.get('hsv_v', 0.4)
        
        img_hsv[..., 0] = (img_hsv[..., 0] + (np.random.random() - 0.5) * 2 * h) % 180
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] * (1 + (np.random.random() - 0.5) * 2 * s), 0, 1)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] * (1 + (np.random.random() - 0.5) * 2 * v), 0, 1)
        
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB).transpose(2, 0, 1)
        return img
    
    @staticmethod
    def collate_fn(batch: List[Tuple]) -> Tuple[np.ndarray, List[np.ndarray], List[str]]:
        """
        批次整理函数
        
        Args:
            batch: 批次数据列表
            
        Returns:
            imgs: 批次图像 (B, C, H, W)
            labels: 批次标签列表
            paths: 图像路径列表
        """
        imgs, labels, paths = zip(*batch)
        
        # 堆叠图像
        imgs = np.stack(imgs, axis=0)
        
        return imgs, list(labels), list(paths)


def create_data_yaml(
    yaml_path: str,
    train_path: str,
    val_path: str,
    nc: int,
    names: List[str],
    test_path: Optional[str] = None
):
    """
    创建YOLO数据集的YAML配置文件
    
    Args:
        yaml_path: YAML文件保存路径
        train_path: 训练图像目录
        val_path: 验证图像目录
        nc: 类别数量
        names: 类别名称列表
        test_path: 测试图像目录(可选)
    """
    data_dict = {
        'train': str(train_path),
        'val': str(val_path),
        'nc': nc,
        'names': names
    }
    
    if test_path:
        data_dict['test'] = str(test_path)
    
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_dict, f, default_flow_style=False, allow_unicode=True)
    
    print(f"数据配置文件已保存到: {yaml_path}")


def verify_dataset(yaml_path: str) -> bool:
    """
    验证数据集是否正确
    
    Args:
        yaml_path: 数据集YAML文件路径
        
    Returns:
        是否验证通过
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data_dict = yaml.safe_load(f)
    
    required_keys = ['train', 'val', 'nc', 'names']
    for key in required_keys:
        if key not in data_dict:
            print(f"错误: 缺少必需的键 '{key}'")
            return False
    
    # 检查类别数量是否匹配
    if len(data_dict['names']) != data_dict['nc']:
        print(f"错误: 类别数量({data_dict['nc']})与名称列表长度({len(data_dict['names'])})不匹配")
        return False
    
    # 检查目录是否存在
    train_path = Path(data_dict['train'])
    val_path = Path(data_dict['val'])
    
    if not train_path.exists():
        print(f"警告: 训练目录不存在: {train_path}")
    
    if not val_path.exists():
        print(f"警告: 验证目录不存在: {val_path}")
    
    print("数据集验证通过!")
    return True