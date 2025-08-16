import os
import logging
import time
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
import threading
import pandas as pd
from config import LOG_FILE, LOG_MAX_BYTES, LOG_BACKUP_COUNT, LOG_FORMAT, LOG_DATE_FORMAT

logger_lock = threading.Lock()


def init_logger(log_file=LOG_FILE, max_bytes=LOG_MAX_BYTES, backup_count=LOG_BACKUP_COUNT):
    """初始化日志系统（线程安全）"""
    logger = logging.getLogger("stock_predict_system")
    logger.setLevel(logging.INFO)

    with logger_lock:
        if logger.handlers:
            return logger

        # 控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 文件handler（带轮转）
        file_handler = None
        try:
            # 确保日志目录存在
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # 尝试创建日志文件
            file_handler = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            file_handler.setLevel(logging.DEBUG)
        except Exception as e:
            # 如果文件日志失败，只使用控制台日志
            print(f"日志文件初始化失败（仅控制台输出）：{str(e)}")
            file_handler = None

        # 日志格式（使用配置文件中的格式）
        formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        console_handler.setFormatter(formatter)
        if file_handler:
            file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        if file_handler:
            logger.addHandler(file_handler)

        # 避免重复输出
        logger.propagate = False
        
        logger.info("日志系统初始化完成")
    return logger


def create_dir(path, exist_ok=True):
    """创建目录（简化错误提示）"""
    if not path:
        raise ValueError("目录路径不能为空")
    try:
        os.makedirs(path, exist_ok=exist_ok)
        return path
    except OSError as e:
        raise OSError(f"创建目录失败：{path} → {str(e)}") from e


def get_current_date(fmt="%Y-%m-%d"):
    """获取当前日期"""
    try:
        return datetime.now().strftime(fmt)
    except ValueError as e:
        raise ValueError(f"日期格式错误：{fmt} → 示例：%Y-%m-%d") from e


def time_it(func):
    """函数耗时装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger = init_logger()
        logger.info(f"函数[{func.__name__}]执行完成，耗时：{end_time - start_time:.2f}秒")
        return result
    return wrapper


def get_stock_name_by_code(stock_code, stock_list_path="stock_list.csv"):
    """
    根据股票代码（如sz.301551）从本地stock_list.csv中获取股票名称
    :param stock_code: 标准股票代码（如sz.301551）
    :param stock_list_path: 本地股票列表路径
    :return: 股票名称（如"无线传媒"），未找到则返回空字符串
    """
    if not stock_code or "." not in stock_code:
        logger = init_logger()
        logger.warning(f"股票代码格式错误：{stock_code}")
        return ""

    try:
        # 检查文件是否存在
        if not os.path.exists(stock_list_path):
            logger = init_logger()
            logger.warning(f"股票列表文件不存在：{stock_list_path}")
            return ""

        # 读取CSV（兼容UTF-8和GBK编码）
        try:
            stock_df = pd.read_csv(stock_list_path, encoding="utf-8")
        except UnicodeDecodeError:
            stock_df = pd.read_csv(stock_list_path, encoding="gbk")

        # 校验列名
        if "code" not in stock_df.columns or "name" not in stock_df.columns:
            logger = init_logger()
            logger.warning(f"{stock_list_path}格式错误，需包含code和name列")
            return ""

        # 匹配代码（精确匹配）
        stock_df["code"] = stock_df["code"].astype(str).str.strip()  # 去重空格
        matched_row = stock_df[stock_df["code"] == stock_code]

        if not matched_row.empty:
            return matched_row["name"].iloc[0].strip()  # 返回第一个匹配的名称
        else:
            logger = init_logger()
            logger.warning(f"未找到{stock_code}对应的股票名称")
            return ""

    except Exception as e:
        logger = init_logger()
        logger.warning(f"获取股票名称失败：{str(e)}")
        return ""


def validate_stock_code(stock_code):
    """
    验证股票代码格式
    :param stock_code: 股票代码
    :return: (is_valid, error_message)
    """
    if not stock_code or not isinstance(stock_code, str):
        return False, "股票代码不能为空"
    
    stock_code = stock_code.strip()
    if "." not in stock_code:
        return False, "代码格式错误！需包含'.'（示例：sz.000858）"
    
    valid_prefixes = ["sh.", "sz."]
    if not any(stock_code.startswith(prefix) for prefix in valid_prefixes):
        return False, f"代码前缀错误！仅支持{valid_prefixes}（示例：sz.000858）"
    
    return True, ""


def safe_filename(filename):
    """
    生成安全的文件名
    :param filename: 原始文件名
    :return: 安全的文件名
    """
    import re
    # 去除或替换不安全的字符
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    return safe_name


def format_file_size(size_bytes):
    """
    格式化文件大小
    :param size_bytes: 字节数
    :return: 格式化的文件大小字符串
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"


def check_dependencies():
    """
    检查必要的依赖包是否已安装
    :return: (is_ready, missing_packages)
    """
    required_packages = [
        'tensorflow', 'pandas', 'numpy', 'matplotlib', 
        'baostock', 'fuzzywuzzy', 'openpyxl', 'sklearn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages