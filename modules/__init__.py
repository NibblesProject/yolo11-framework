import pkgutil
import importlib
import inspect
import torch.nn as nn
from pathlib import Path

# 自动扫描当前文件夹下的所有 .py 文件
for loader, module_name, is_pkg in pkgutil.iter_modules([str(Path(__file__).parent)]):
    # 导入模块
    module = importlib.import_module(f'.{module_name}', package=__name__)
    # 将模块中的 nn.Module 子类暴露到当前包的命名空间
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, nn.Module) and obj is not nn.Module:
            globals()[name] = obj