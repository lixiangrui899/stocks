import os
import time
import random
import baostock as bs
import pandas as pd
from datetime import datetime, timedelta
from utils import init_logger, create_dir, validate_stock_code
from config import (
    DATA_DIR, CRAWL_DAYS, CRAWL_FIELDS, ADJUST_FLAG, 
    MAX_RETRY, RETRY_DELAY, VALID_MARKETS
)
import numpy as np

logger = init_logger()


def crawl_stock_data(stock_code, days=CRAWL_DAYS):
    """爬取股票历史数据（后复权，适配模型特征需求）"""
    # 1. 代码校验
    is_valid, error_msg = validate_stock_code(stock_code)
    if not is_valid:
        raise ValueError(error_msg)

    # 2. 时间范围
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    logger.info(f"爬取 {stock_code} 数据（{start_date} → {end_date}）")

    # 3. 多轮重试爬取
    for retry in range(MAX_RETRY):
        lg = None
        try:
            lg = bs.login()
            if lg.error_code != "0":
                raise ValueError(f"Baostock登录失败：{lg.error_msg}（错误码：{lg.error_code}）")

            # 调用接口
            rs = bs.query_history_k_data_plus(
                stock_code,
                "date,code,open,high,low,close,volume,amount,adjustflag",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag=ADJUST_FLAG  # 后复权（必须设置！否则除权导致价格断层）
            )
            if rs.error_code != "0":
                raise ValueError(f"爬取失败：{rs.error_msg}（错误码：{rs.error_code}）")

            # 解析数据
            df = rs.get_data()
            bs.logout()
            if df.empty:
                raise ValueError("接口返回空数据（可能是非交易日或代码错误）")
            break

        except Exception as e:
            if lg:
                try:
                    bs.logout()
                except:
                    pass
            if retry == MAX_RETRY - 1:
                raise ValueError(f"{MAX_RETRY}次重试失败：{str(e)}\n建议：检查网络/代码有效性")
            wait_time = RETRY_DELAY + retry * 5 + random.random() * 3
            logger.warning(f"第{retry+1}次爬取失败：{str(e)[:50]}...，{wait_time:.1f}秒后重试")
            time.sleep(wait_time)

    # 4. 数据清洗
    try:
        df["date"] = pd.to_datetime(df["date"])
        numeric_cols = ["open", "close", "low", "high", "volume", "amount"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 过滤无效数据
        df = df.dropna(subset=numeric_cols)
        df = df[df["volume"] > 0].sort_values("date").reset_index(drop=True)

        if len(df) < 30:
            raise ValueError(f"有效数据仅{len(df)}条（需≥30条才能建模）")
        logger.info(f"数据清洗完成：{len(df)}条有效记录")

    except Exception as e:
        raise ValueError(f"数据清洗失败：{str(e)}") from e

    # 5. 保存数据
    try:
        create_dir(DATA_DIR)
        filename = f"{stock_code.replace('.', '')}_history.csv"
        save_path = os.path.join(DATA_DIR, filename)
        df.to_csv(save_path, index=False, encoding="utf-8")
        logger.info(f"数据已保存至：{save_path}")
    except Exception as e:
        raise ValueError(f"数据保存失败：{str(e)}") from e

    return df, save_path


def validate_crawled_data(df):
    """
    验证爬取的数据质量
    :param df: 爬取的数据框
    :return: (is_valid, issues)
    """
    issues = []
    
    # 检查数据量
    if len(df) < 30:
        issues.append(f"数据量不足：仅{len(df)}条记录，建议≥30条")
    
    # 检查时间跨度
    if len(df) > 0:
        date_range = (df['date'].max() - df['date'].min()).days
        if date_range < 30:
            issues.append(f"时间跨度不足：仅{date_range}天，建议≥30天")
    
    # 检查缺失值
    missing_counts = df.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            issues.append(f"列'{col}'有{count}个缺失值")
    
    # 检查异常值
    numeric_cols = ['open', 'close', 'low', 'high', 'volume']
    for col in numeric_cols:
        if col in df.columns:
            # 检查负值
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                issues.append(f"列'{col}'有{negative_count}个负值")
            
            # 检查零值（除了成交量）
            if col != 'volume':
                zero_count = (df[col] == 0).sum()
                if zero_count > 0:
                    issues.append(f"列'{col}'有{zero_count}个零值")
    
    return len(issues) == 0, issues


def get_data_summary(df):
    """
    获取数据摘要信息
    :param df: 数据框
    :return: 摘要信息字典
    """
    if df.empty:
        return {"error": "数据为空"}
    
    summary = {
        "总记录数": len(df),
        "时间范围": f"{df['date'].min().strftime('%Y-%m-%d')} 至 {df['date'].max().strftime('%Y-%m-%d')}",
        "时间跨度": f"{(df['date'].max() - df['date'].min()).days}天",
        "特征列": list(df.columns)
    }
    
    # 数值列统计
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in df.columns:
            summary[f"{col}_均值"] = round(df[col].mean(), 2)
            summary[f"{col}_最大值"] = round(df[col].max(), 2)
            summary[f"{col}_最小值"] = round(df[col].min(), 2)
    
    return summary