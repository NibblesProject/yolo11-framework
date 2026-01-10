"""
推理模块
提供YOLO11模型的推理功能
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from PIL import Image
import json

from model import YOLO11Model


class YOLO11Inference:
    """YOLO11推理器"""
    
    def __init__(
        self,
        model_path: str,
        device: str = '0',
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 300
    ):
        """
        初始化推理器
        
        Args:
            model_path: 模型权重路径
            device: 设备
            conf: 置信度阈值
            iou: IoU阈值
            max_det: 每张图最大检测数
        """
        self.model = YOLO11Model(
            weights=model_path,
            device=device,
            verbose=False
        )
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        
        # 获取类别名称
        self.class_names = self.model.model.names if hasattr(self.model.model, 'names') else None
        
        print(f"推理器初始化完成: {model_path}")
        if self.class_names:
            print(f"类别数量: {len(self.class_names)}")
    
    def predict(
        self,
        source: Union[str, np.ndarray, List],
        save: bool = False,
        save_dir: Optional[str] = None,
        show: bool = False,
        stream: bool = False
    ):
        """
        推理
        
        Args:
            source: 输入源 (图像路径/视频路径/目录/URL/numpy数组)
            save: 是否保存结果
            save_dir: 保存目录
            show: 是否显示结果
            stream: 是否流式输出
            
        Returns:
            推理结果
        """
        results = self.model.predict(
            source=source,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            save=save,
            show=show,
            stream=stream,
            verbose=False
        )
        
        return results
    
    def predict_image(self, image_path: str) -> Dict[str, Any]:
        """
        对单张图像进行推理
        
        Args:
            image_path: 图像路径
            
        Returns:
            推理结果字典
        """
        results = self.predict(image_path, stream=False)
        
        # 解析结果
        result = self._parse_results(results)
        result['image_path'] = str(image_path)
        
        return result
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        批量推理
        
        Args:
            image_paths: 图像路径列表
            
        Returns:
            推理结果列表
        """
        results = []
        for image_path in image_paths:
            result = self.predict_image(image_path)
            results.append(result)
        
        return results
    
    def predict_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show: bool = False,
        save_video: bool = False
    ):
        """
        视频推理
        
        Args:
            video_path: 视频路径
            output_path: 输出视频路径
            show: 是否显示结果
            save_video: 是否保存视频
            
        Returns:
            帧结果列表
        """
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 创建视频写入器
        video_writer = None
        if save_video and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 处理每一帧
        frame_results = []
        frame_count = 0
        
        print(f"处理视频: {video_path}")
        print(f"总帧数: {total_frames}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 推理
            results = self.predict(frame, stream=False)
            
            # 解析结果
            result = self._parse_results(results)
            result['frame_index'] = frame_count
            frame_results.append(result)
            
            # 绘制结果
            annotated_frame = self._draw_results(frame, result['detections'])
            
            # 显示
            if show:
                cv2.imshow('YOLO11 Inference', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 保存视频
            if video_writer:
                video_writer.write(annotated_frame)
            
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"已处理 {frame_count}/{total_frames} 帧")
        
        # 释放资源
        cap.release()
        if video_writer:
            video_writer.release()
        if show:
            cv2.destroyAllWindows()
        
        print(f"视频处理完成! 共处理 {frame_count} 帧")
        
        return frame_results
    
    def _parse_results(self, results) -> Dict[str, Any]:
        """
        解析推理结果
        
        Args:
            results: Ultralytics推理结果
            
        Returns:
            解析后的结果字典
        """
        detections = []
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    # 获取边界框
                    box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = boxes.conf[i].cpu().numpy()  # 置信度
                    cls = int(boxes.cls[i].cpu().numpy())  # 类别
                    
                    detection = {
                        'bbox': box.tolist(),
                        'confidence': float(conf),
                        'class_id': cls,
                        'class_name': self.class_names[cls] if self.class_names and cls < len(self.class_names) else str(cls)
                    }
                    detections.append(detection)
        
        return {
            'detections': detections,
            'num_detections': len(detections)
        }
    
    def _draw_results(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        line_thickness: int = 2
    ) -> np.ndarray:
        """
        绘制检测结果
        
        Args:
            image: 原始图像
            detections: 检测结果列表
            line_thickness: 线条粗细
            
        Returns:
            绘制后的图像
        """
        img = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']
            
            # 绘制边界框
            x1, y1, x2, y2 = map(int, bbox)
            color = self._get_color(det['class_id'])
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)
            
            # 绘制标签
            label = f"{class_name}: {conf:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            cv2.rectangle(
                img,
                (x1, y1 - label_height - baseline),
                (x1 + label_width, y1),
                color,
                -1
            )
            
            cv2.putText(
                img,
                label,
                (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return img
    
    def _get_color(self, class_id: int) -> tuple:
        """
        根据类别ID获取颜色
        
        Args:
            class_id: 类别ID
            
        Returns:
            颜色 (B, G, R)
        """
        # 使用固定的颜色映射
        colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 洋红
            (0, 255, 255),  # 黄色
            (128, 0, 128),  # 紫色
            (255, 165, 0),  # 橙色
            (0, 128, 128),  # 深青色
            (128, 128, 0),  # 橄榄色
        ]
        
        return colors[class_id % len(colors)]
    
    def save_results(
        self,
        results: List[Dict[str, Any]],
        save_path: str,
        format: str = 'json'
    ):
        """
        保存推理结果
        
        Args:
            results: 推理结果列表
            save_path: 保存路径
            format: 保存格式 ('json', 'txt')
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"结果已保存到: {save_path}")
        
        elif format == 'txt':
            with open(save_path, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(f"Image: {result.get('image_path', 'unknown')}\n")
                    f.write(f"Detections: {result['num_detections']}\n")
                    for det in result['detections']:
                        bbox = det['bbox']
                        f.write(
                            f"{det['class_id']} {bbox[0]} {bbox[1]} "
                            f"{bbox[2]} {bbox[3]} {det['confidence']:.4f}\n"
                        )
                    f.write("\n")
            print(f"结果已保存到: {save_path}")
        
        else:
            raise ValueError(f"不支持的格式: {format}")


def predict_image_simple(
    model_path: str,
    image_path: str,
    conf: float = 0.25,
    device: str = '0'
) -> Dict[str, Any]:
    """
    简单图像推理函数
    
    Args:
        model_path: 模型路径
        image_path: 图像路径
        conf: 置信度阈值
        device: 设备
        
    Returns:
        推理结果
    """
    inferencer = YOLO11Inference(model_path, device=device, conf=conf)
    return inferencer.predict_image(image_path)


def predict_video_simple(
    model_path: str,
    video_path: str,
    output_path: str,
    conf: float = 0.25,
    device: str = '0'
):
    """
    简单视频推理函数
    
    Args:
        model_path: 模型路径
        video_path: 视频路径
        output_path: 输出视频路径
        conf: 置信度阈值
        device: 设备
    """
    inferencer = YOLO11Inference(model_path, device=device, conf=conf)
    return inferencer.predict_video(video_path, output_path=output_path, save_video=True)