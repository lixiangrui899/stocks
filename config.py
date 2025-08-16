"""
股票预测系统配置文件
统一管理所有模块的配置项，避免重复定义
"""

import os
from datetime import datetime
import sys # Added for sys.executable

# ==================== 基础配置 ====================
# 项目根目录 - 修复exe环境中的路径问题
if getattr(sys, 'frozen', False):
    # 如果是exe运行，使用exe所在目录
    PROJECT_ROOT = os.path.dirname(sys.executable)
else:
    # 如果是Python脚本运行，使用脚本所在目录
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据目录 - 使用绝对路径
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "figures")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")

# 文件路径
STOCK_LIST_PATH = os.path.join(PROJECT_ROOT, "stock_list.csv")
LOG_FILE = os.path.join(PROJECT_ROOT, "app.log")

# ==================== 模型配置 ====================
# 特征列（必须与训练和预测保持一致）
FEATURE_COLUMNS = ["close", "high", "low", "volume"]

# 模型参数
LOOK_BACK = 60  # 时间步长
EPOCHS = 50     # 训练轮数
BATCH_SIZE = 32 # 批次大小

# 模型文件路径
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_stock_model.keras")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_predict_model.keras")

# ==================== 数据爬取配置 ====================
# 爬取参数
CRAWL_DAYS = 6000  # 默认爬取天数
CRAWL_FIELDS = "date,open,close,low,high,volume,amount"
ADJUST_FLAG = "3"  # 后复权

# 重试配置
MAX_RETRY = 3
RETRY_DELAY = 5

# 有效市场前缀
VALID_MARKETS = ["sh.", "sz."]

# ==================== 预测配置 ====================
# 预测天数
PREDICTION_DAYS = 30

# 预测质量参数
MAX_PRICE_JUMP_RATIO = 0.05  # 最大价格跳跃比例（5%）
HISTORICAL_RANGE_MARGIN = 0.2  # 历史价格范围边界（20%）
VOLATILITY_FACTOR = 0.5  # 波动性因子
MIN_VOLATILITY_RATIO = 0.1  # 最小波动性比例
MAX_VOLATILITY_RATIO = 3.0  # 最大波动性比例
TREND_DIFFERENCE_THRESHOLD = 0.1  # 趋势差异阈值（10%）

# 图表配置
CHART_DPI = 300
CHART_FIGSIZE = (12, 7)

# ==================== GUI配置 ====================
# 窗口配置
GUI_WIDTH = 650
GUI_HEIGHT = 350
GUI_TITLE = "股票预测系统（未来30天收盘价）"

# 搜索配置
FUZZY_MATCH_THRESHOLD = 60  # 模糊匹配阈值
SIMILAR_MATCH_THRESHOLD = 50  # 相似匹配阈值

# ==================== 日志配置 ====================
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5
LOG_FORMAT = '[%(asctime)s] [线程:%(thread)d] [%(levelname)s] %(message)s'
LOG_DATE_FORMAT = '%Y/%m/%d %H:%M:%S'

# ==================== 工具函数 ====================
def get_current_date(fmt="%Y-%m-%d"):
    """获取当前日期"""
    return datetime.now().strftime(fmt)

def get_safe_filename(filename):
    """生成安全的文件名（去除特殊字符）"""
    import re
    # 去除或替换不安全的字符
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    return safe_name

def ensure_dir_exists(path):
    """确保目录存在，不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

# ==================== 初始化目录 ====================
def init_directories():
    """初始化所有必要的目录"""
    directories = [DATA_DIR, MODEL_DIR, FIGURE_DIR, RESULT_DIR]
    for directory in directories:
        ensure_dir_exists(directory)

# 在模块导入时自动初始化目录
init_directories()
