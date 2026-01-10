#!/usr/bin/env python3
"""
YOLO11 训练前测试检查脚本
整合配置、数据集和模型的完整检查
"""
import re
from pathlib import Path
from config import Config, get_config
from model import YOLO11Model
from dataset import verify_dataset


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
    UNDERLINE = '\033[4m'


def get_display_width(text: str) -> int:
    """
    计算字符串在终端的显示宽度
    
    正确处理中英文混合字符：
    - 英文字符、数字、半角符号：1个显示宽度
    - 中文字符、全角符号：2个显示宽度
    - ANSI颜色代码：不计入显示宽度
    
    Args:
        text: 要计算的字符串
    
    Returns:
        显示宽度
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


def pad_text(text: str, width: int, align: str = 'left') -> str:
    """
    填充文本到指定显示宽度
    
    Args:
        text: 原始文本
        width: 目标显示宽度
        align: 对齐方式（left, right, center）
    
    Returns:
        填充后的文本
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


def print_box(title: str, content: list = None, width: int = 80, color: str = None):
    """
    打印带边框的文本框
    
    Args:
        title: 标题
        content: 内容行列表
        width: 框的宽度
        color: 颜色
    """
    if color:
        print(color)
    
    # 上边框
    print('╔' + '═' * (width - 2) + '╗')
    
    # 标题行
    title_line = f'║ {pad_text(title, width - 3)} ║'
    print(title_line)
    
    # 分隔线
    print('╠' + '═' * (width - 2) + '╣')
    
    # 内容
    if content:
        for line in content:
            content_line = f'║ {pad_text(line, width - 3)} ║'
            print(content_line)
    
    # 下边框
    print('╚' + '═' * (width - 2) + '╝')
    
    if color:
        print(Colors.ENDC)


def print_status(message: str, status: str = 'info'):
    """
    打印带状态的消息
    
    Args:
        message: 消息内容
        status: 状态（info, success, warning, error）
    """
    color_map = {
        'info': Colors.OKBLUE,
        'success': Colors.OKGREEN,
        'warning': Colors.WARNING,
        'error': Colors.FAIL
    }
    icon_map = {
        'info': 'ℹ',
        'success': '✓',
        'warning': '⚠',
        'error': '✗'
    }
    
    color = color_map.get(status, Colors.OKBLUE)
    icon = icon_map.get(status, 'ℹ')
    
    print(f"{color}{icon}{Colors.ENDC} {message}")


def check_data_yaml():
    """检查data.yaml配置"""
    print_box("数据集配置检查 (data.yaml)")
    
    try:
        config = get_config()
        
        checks = []
        
        # 检查类别名称
        if config.data.names:
            print_status(f"类别名称: {', '.join(config.data.names)}", 'success')
            checks.append(f"  类别名称: {', '.join(config.data.names)}")
        else:
            print_status("未找到类别名称", 'error')
            checks.append("  类别名称: 未找到")
        
        # 检查类别数量
        if config.data.nc == 6:
            print_status(f"类别数量: {config.data.nc}", 'success')
            checks.append(f"  类别数量: {config.data.nc}")
        else:
            print_status(f"类别数量: {config.data.nc} (期望: 6)", 'warning')
            checks.append(f"  类别数量: {config.data.nc} (期望: 6)")
        
        # 检查训练路径
        train_path = Path(config.data.train_path)
        if train_path.exists():
            # 统计图像数量
            images = list(train_path.glob("*.jpg")) + list(train_path.glob("*.png"))
            print_status(f"训练路径: {config.data.train_path} ({len(images)} 张图像)", 'success')
            checks.append(f"  训练路径: {config.data.train_path} ({len(images)} 张图像)")
        else:
            print_status(f"训练路径不存在: {config.data.train_path}", 'error')
            checks.append(f"  训练路径: 不存在")
        
        # 检查验证路径
        val_path = Path(config.data.val_path)
        if val_path.exists():
            images = list(val_path.glob("*.jpg")) + list(val_path.glob("*.png"))
            print_status(f"验证路径: {config.data.val_path} ({len(images)} 张图像)", 'success')
            checks.append(f"  验证路径: {config.data.val_path} ({len(images)} 张图像)")
        else:
            print_status(f"验证路径不存在: {config.data.val_path}", 'error')
            checks.append(f"  验证路径: 不存在")
        
        print()
        return True, config
    except Exception as e:
        print_status(f"加载配置失败: {e}", 'error')
        print()
        return False, None


def check_model_loading(config: Config):
    """检查模型加载"""
    print_box("模型加载检查")
    
    try:
        # 检查本地模型
        local_model_dir = Path(config.model.local_model_dir)
        local_model_path = local_model_dir / f"{config.model.model_name}.pt"
        
        checks = []
        
        if local_model_path.exists():
            size_mb = local_model_path.stat().st_size / 1024 / 1024
            print_status(f"本地模型: {local_model_path} ({size_mb:.2f} MB)", 'success')
            checks.append(f"  本地模型: {local_model_path} ({size_mb:.2f} MB)")
        else:
            print_status(f"本地模型不存在，将下载: {config.model.model_name}", 'warning')
            checks.append(f"  本地模型: 不存在，将下载")
        
        # 尝试加载模型
        print_status("尝试加载模型...", 'info')
        model = YOLO11Model(
            model_name=config.model.model_name,
            pretrained=config.model.pretrained,
            weights=config.model.weights,
            device='cpu',  # 测试时使用CPU
            verbose=False,
            local_model_dir=config.model.local_model_dir
        )
        
        print_status("模型加载成功!", 'success')
        
        # 显示模型信息
        info = model.get_model_info()
        print_status(f"模型名称: {info.get('model_name', 'N/A')}", 'info')
        print_status(f"参数数量: {info.get('parameters', 'N/A')}", 'info')
        print_status(f"测试设备: {info.get('device', 'N/A')} (训练时将使用GPU)", 'info')
        
        # 说明类别信息
        num_classes = len(info.get('classes', {}))
        print_status(f"预训练类别数: {num_classes} (COCO数据集)", 'info')
        print_status(f"训练将使用: {config.data.nc} 个类别 ({', '.join(config.data.names[:3])}...)", 'success')
        
        print()
        return True
    except Exception as e:
        print_status(f"模型加载失败: {e}", 'error')
        print()
        return False


def check_dataset():
    """检查数据集完整性"""
    print_box("数据集完整性检查")
    
    try:
        data_yaml_path = Path("data/data.yaml")
        if not data_yaml_path.exists():
            print_status("data.yaml不存在", 'error')
            print()
            return False
        
        result = verify_dataset(str(data_yaml_path))
        
        if result:
            print_status("数据集验证通过", 'success')
            print()
            return True
        else:
            print_status("数据集验证失败", 'error')
            print()
            return False
    except Exception as e:
        print_status(f"数据集检查失败: {e}", 'error')
        print()
        return False


def check_system():
    """检查系统环境"""
    print_box("系统环境检查")
    
    checks = []
    
    # 检查Python版本
    import sys
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print_status(f"Python版本: {py_version}", 'info')
    checks.append(f"  Python: {py_version}")
    
    # 检查PyTorch
    try:
        import torch
        print_status(f"PyTorch版本: {torch.__version__}", 'info')
        checks.append(f"  PyTorch: {torch.__version__}")
        
        # 检查CUDA
        if torch.cuda.is_available():
            print_status(f"CUDA可用: 是 (版本: {torch.version.cuda})", 'success')
            print_status(f"GPU数量: {torch.cuda.device_count()}", 'info')
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print_status(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.2f} GB)", 'info')
                checks.append(f"  GPU {i}: {props.name}")
        else:
            print_status("CUDA不可用，将使用CPU", 'warning')
            checks.append(f"  CUDA: 不可用")
    except ImportError:
        print_status("PyTorch未安装", 'error')
        checks.append("  PyTorch: 未安装")
    
    # 检查ultralytics
    try:
        from ultralytics import __version__
        print_status(f"Ultralytics版本: {__version__}", 'info')
        checks.append(f"  Ultralytics: {__version__}")
    except ImportError:
        print_status("Ultralytics未安装", 'error')
        checks.append("  Ultralytics: 未安装")
    
    print()
    return len([c for c in checks if '错误' in c or '失败' in c]) == 0


def run_all_tests():
    """运行所有测试"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'YOLO11 训练前检查'}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")
    
    results = []
    
    # 系统环境检查
    sys_ok = check_system()
    results.append(('系统环境', sys_ok))
    
    # 数据配置检查
    data_ok, config = check_data_yaml()
    results.append(('数据配置', data_ok))
    
    # 数据集完整性检查
    if data_ok:
        dataset_ok = check_dataset()
        results.append(('数据集完整性', dataset_ok))
    else:
        dataset_ok = False
        results.append(('数据集完整性', False))
    
    # 模型加载检查
    if data_ok and config:
        model_ok = check_model_loading(config)
        results.append(('模型加载', model_ok))
    else:
        model_ok = False
        results.append(('模型加载', False))
    
    # 打印总结
    print_box("检查总结")
    
    all_ok = True
    for name, ok in results:
        status = Colors.OKGREEN + "✓ 通过" + Colors.ENDC if ok else Colors.FAIL + "✗ 失败" + Colors.ENDC
        print(f"{name:20s} {status}")
        if not ok:
            all_ok = False
    
    print()
    
    if all_ok:
        print(f"{Colors.OKGREEN}{Colors.BOLD}✓ 所有检查通过，可以开始训练！{Colors.ENDC}\n")
        return True
    else:
        print(f"{Colors.FAIL}{Colors.BOLD}✗ 部分检查失败，请修复后再开始训练！{Colors.ENDC}\n")
        return False


if __name__ == '__main__':
    run_all_tests()
