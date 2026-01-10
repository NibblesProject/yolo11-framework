#!/usr/bin/env python3
"""
YOLO11训练脚本
主入口文件，提供命令行接口进行训练
"""
import argparse
import sys
import os
import subprocess
import re
from pathlib import Path

from config import Config, get_config
from trainer import YOLO11Trainer, train_simple, train_from_config
from utils import set_seed, check_gpu


class Colors:
    """ANSI颜色代码"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def get_display_width(text: str) -> int:
    """
    计算字符串在终端的显示宽度
    
    正确处理中英文混合字符：
    - 英文字符、数字、半角符号：1个显示宽度
    - 中文字符、全角符号：2个显示宽度
    - ANSI颜色代码：不计入显示宽度
    """
    # ANSI颜色代码正则表达式
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    
    # 移除ANSI颜色代码
    clean_text = ansi_escape.sub('', text)
    
    width = 0
    for char in clean_text:
        # 获取字符的Unicode码点
        code = ord(char)
        
        # 判断字符类型
        # 中文、日文、韩文等全角字符：2个显示宽度
        if (
            (0x4E00 <= code <= 0x9FFF) or  # CJK统一汉字
            (0x3400 <= code <= 0x4DBF) or  # CJK扩展A
            (0x20000 <= code <= 0x2A6DF) or  # CJK扩展B
            (0x2A700 <= code <= 0x2B73F) or  # CJK扩展C
            (0x2B740 <= code <= 0x2B81F) or  # CJK扩展D
            (0x2B820 <= code <= 0x2CEAF) or  # CJK扩展E
            (0x2CEB0 <= code <= 0x2EBEF) or  # CJK扩展F
            (0x3000 <= code <= 0x303F) or  # CJK符号和标点
            (0xFF00 <= code <= 0xFFEF) or  # 半角及全角形式
            (0x1100 <= code <= 0x11FF) or  # 韩文字母
            (0xAC00 <= code <= 0xD7AF) or  # 韩文音节
            (0x3040 <= code <= 0x30FF) or  # 日文平假名和片假名
            (0x31F0 <= code <= 0x31FF) or  # 日文扩展
            (0x3200 <= code <= 0x32FF)    # 中文符号
        ):
            width += 2
        else:
            # ASCII字符、数字、半角符号：1个显示宽度
            width += 1
    
    return width


def pad_text(text: str, width: int, align: str = 'center') -> str:
    """
    填充文本到指定显示宽度
    
    Args:
        text: 原始文本
        width: 目标显示宽度
        align: 对齐方式（left, right, center）
    """
    display_width = get_display_width(text)
    padding = width - display_width
    
    if padding < 0:
        return text
    
    if align == 'right':
        return ' ' * padding + text
    elif align == 'center':
        left_pad = padding // 2
        right_pad = padding - left_pad
        return ' ' * left_pad + text + ' ' * right_pad
    else:  # left
        return text + ' ' * padding


def print_header(title: str):
    """打印标题"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{pad_text(title, 80, 'center')}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")


def print_section(title: str):
    """打印分节标题"""
    print(f"\n{Colors.OKBLUE}{Colors.BOLD}{'─' * 80}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{Colors.BOLD}{title}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{Colors.BOLD}{'─' * 80}{Colors.ENDC}\n")


def print_info(message: str):
    """打印信息"""
    print(f"{Colors.OKBLUE}ℹ{Colors.ENDC} {message}")


def print_success(message: str):
    """打印成功消息"""
    print(f"{Colors.OKGREEN}✓{Colors.ENDC} {message}")


def print_warning(message: str):
    """打印警告"""
    print(f"{Colors.WARNING}⚠{Colors.ENDC} {message}")


def print_error(message: str):
    """打印错误"""
    print(f"{Colors.FAIL}✗{Colors.ENDC} {message}")


def run_pre_check():
    """运行训练前检查"""
    print_header("运行训练前检查")
    
    try:
        from test import run_all_tests
        return run_all_tests()
    except Exception as e:
        print_error(f"检查过程出错: {e}")
        return False


def start_monitoring(exp_dir: Path):
    """
    启动监控进程（TensorBoard和nvidia-smi）
    
    Args:
        exp_dir: 实验目录
    """
    processes = []
    
    # 启动TensorBoard
    try:
        logs_dir = exp_dir / "logs"
        if logs_dir.exists():
            print_info("启动TensorBoard...")
            tensorboard_cmd = ["tensorboard", "--logdir", str(logs_dir), "--port", "6006"]
            tb_process = subprocess.Popen(
                tensorboard_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            processes.append(("TensorBoard", tb_process))
            print_success(f"TensorBoard已启动，访问地址: http://localhost:6006")
    except Exception as e:
        print_warning(f"启动TensorBoard失败: {e}")
    
    # 启动nvidia-smi监控（仅在有CUDA时）
    try:
        import torch
        if torch.cuda.is_available():
            print_info("启动GPU监控...")
            nvidia_cmd = ["watch", "-n", "1", "nvidia-smi"]
            smi_process = subprocess.Popen(
                nvidia_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            processes.append(("nvidia-smi", smi_process))
            print_success("GPU监控已启动（按Ctrl+C停止监控）")
    except Exception as e:
        print_warning(f"启动GPU监控失败: {e}")
    
    return processes


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='YOLO11训练脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基本参数
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径 (YAML格式)')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'resume', 'validate', 'export'],
                        help='运行模式')
    
    # 数据参数
    parser.add_argument('--data', type=str, default='data/data.yaml',
                        help='数据集配置文件路径')
    parser.add_argument('--train-path', type=str, default=None,
                        help='训练数据路径')
    parser.add_argument('--val-path', type=str, default=None,
                        help='验证数据路径')
    parser.add_argument('--nc', type=int, default=None,
                        help='类别数量')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='yolo11n',
                        choices=['yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x'],
                        help='模型大小')
    parser.add_argument('--weights', type=str, default=None,
                        help='预训练权重路径')
    parser.add_argument('--device', type=str, default='0',
                        help='设备 (0, 1, cpu, cuda:0, etc.)')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='不使用预训练权重')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='图像大小')
    parser.add_argument('--patience', type=int, default=50,
                        help='早停轮数')
    parser.add_argument('--optimizer', type=str, default='auto',
                        choices=['auto', 'SGD', 'Adam', 'AdamW'],
                        help='优化器')
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='初始学习率')
    parser.add_argument('--lrf', type=float, default=0.01,
                        help='最终学习率')
    
    # 保存参数
    parser.add_argument('--project', type=str, default='yolo11',
                        help='项目名称')
    parser.add_argument('--name', type=str, default='exp',
                        help='实验名称')
    parser.add_argument('--save-dir', type=str, default='runs/train',
                        help='保存目录')
    parser.add_argument('--exist-ok', action='store_true',
                        help='覆盖已有目录')
    parser.add_argument('--resume-path', type=str, default=None,
                        help='恢复训练的检查点路径')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--workers', type=int, default=8,
                        help='数据加载线程数')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='使用混合精度训练')
    
    # 导出参数
    parser.add_argument('--export-format', type=str, default='onnx',
                        choices=['onnx', 'torchscript', 'engine', 'coreml', 'saved_model', 'pb', 'tflite'],
                        help='导出格式')
    
    # 测试和监控参数
    parser.add_argument('--no-check', action='store_true',
                        help='跳过训练前检查')
    parser.add_argument('--monitor', action='store_true',
                        help='训练后启动TensorBoard和GPU监控')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    print_header("YOLO11 训练框架")
    
    # 运行训练前检查（train模式）
    if args.mode == 'train' and args.no_check is False:
        print_info("运行训练前环境检查...")
        if not run_pre_check():
            print_error("训练前检查失败，请修复问题后再试！")
            sys.exit(1)
        print("\n")
    
    # 设置随机种子
    print_section("初始化环境")
    print_info(f"设置随机种子: {args.seed}")
    set_seed(args.seed)
    
    # 检查GPU
    check_gpu()
    
    # 加载配置
    if args.config and Path(args.config).exists():
        print_info(f"从配置文件加载: {args.config}")
        config = Config.from_yaml(args.config)
    else:
        config = get_config()
        print_info("使用默认配置")
    
    # 更新配置
    if args.train_path:
        config.data.train_path = args.train_path
    if args.val_path:
        config.data.val_path = args.val_path
    if args.nc:
        config.data.nc = args.nc
    
    config.model.model_name = args.model
    config.model.device = args.device
    if args.weights:
        config.model.weights = args.weights
    if args.no_pretrained:
        config.model.pretrained = False
    
    config.train.epochs = args.epochs
    config.data.batch_size = args.batch_size
    config.data.img_size = args.imgsz
    config.train.patience = args.patience
    config.train.optimizer = args.optimizer
    config.train.lr0 = args.lr0
    config.train.lrf = args.lrf
    config.train.project = args.project
    config.train.name = args.name
    config.train.save_dir = args.save_dir
    config.train.exist_ok = args.exist_ok
    config.data.workers = args.workers
    config.train.amp = args.amp
    
    # 打印配置摘要
    print_section("训练配置")
    print_info(f"模型: {config.model.model_name}")
    print_info(f"设备: {config.model.device}")
    print_info(f"训练轮数: {config.train.epochs}")
    print_info(f"批次大小: {config.data.batch_size}")
    print_info(f"图像大小: {config.data.img_size}")
    print_info(f"初始学习率: {config.train.lr0}")
    print_info(f"优化器: {config.train.optimizer}")
    print_info(f"类别数量: {config.data.nc}")
    print_info(f"类别名称: {', '.join(config.data.names)}")
    
    # 创建训练器
    print_section("初始化训练器")
    trainer = YOLO11Trainer(config=config)
    
    # 监控进程列表
    monitor_processes = []
    
    try:
        # 根据模式执行
        if args.mode == 'train':
            print_section("开始训练")
            trainer.train()
            
            # 训练完成后启动监控
            if args.monitor:
                print_section("启动监控")
                # 获取最新的实验目录
                exp_dirs = sorted(Path("experiments").glob("exp_*"), reverse=True)
                if exp_dirs:
                    exp_dir = exp_dirs[0]
                    monitor_processes = start_monitoring(exp_dir)
                    print_info(f"监控已启动，实验目录: {exp_dir}")
                    print_info("按Ctrl+C停止监控")
                    try:
                        while True:
                            pass
                    except KeyboardInterrupt:
                        print("\n")
                        print_info("停止监控...")
        
        elif args.mode == 'resume':
            if args.resume_path is None:
                print_error("恢复训练需要指定 --resume-path")
                sys.exit(1)
            print_section("恢复训练")
            print_info(f"从检查点恢复训练: {args.resume_path}")
            trainer.resume(args.resume_path)
        
        elif args.mode == 'validate':
            print_section("开始验证")
            results = trainer.validate(args.data)
            print_success("验证完成!")
            print_info(f"mAP50: {results.box.map50:.4f}")
            print_info(f"mAP50-95: {results.box.map:.4f}")
        
        elif args.mode == 'export':
            print_section("导出模型")
            print_info(f"导出模型为 {args.export_format} 格式...")
            trainer.export_model(format=args.export_format)
            print_success("导出完成!")
        
        print_header("任务完成")
        
    except KeyboardInterrupt:
        print_warning("\n训练被用户中断")
    except Exception as e:
        print_error(f"训练出错: {e}")
        raise e
    finally:
        # 清理监控进程
        if monitor_processes:
            print_info("清理监控进程...")
            for name, process in monitor_processes:
                try:
                    process.terminate()
                    print_success(f"已终止 {name}")
                except Exception as e:
                    print_warning(f"终止 {name} 失败: {e}")


if __name__ == '__main__':
    main()