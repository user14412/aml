import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(log_file_dir="log/"):
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"run_{timestamp}.log"
    log_file_path = Path(log_file_dir) / log_file_name

    # 创建一个专属 Logger
    logger = logging.getLogger("AMLLogger")
    logger.setLevel(logging.DEBUG) # 捕捉所有级别的日志

    # 1. 配置文件输出 (FileHandler) - 记录所有细节
    # 确保日志目录存在，如果不存在，静默创建
    Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    # 详尽的格式：时间 - 模块 - 级别 - 信息
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
    file_handler.setFormatter(file_formatter)

    # 2. 配置终端输出 (StreamHandler) - 保持终端清爽
    console_handler = logging.StreamHandler(sys.stdout)
    # 终端只打印 WARNING 及以上（或者你觉得有必要的 INFO）
    console_handler.setLevel(logging.WARNING) 
    # 精简的格式
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
    console_handler.setFormatter(console_formatter)

    # 避免重复添加 Handler
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# 全局单例使用
logger = setup_logger("log/execution/")