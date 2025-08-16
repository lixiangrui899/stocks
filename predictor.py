# predictor.py 完整修改代码（关键部分已标注）
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from datetime import timedelta, datetime
from tensorflow.keras.models import load_model
from utils import init_logger, create_dir, get_stock_name_by_code, safe_filename
from config import (
    RESULT_DIR, FEATURE_COLUMNS, LOOK_BACK, PREDICTION_DAYS,
    LSTM_MODEL_PATH, CHART_DPI, CHART_FIGSIZE
)

# 字体配置
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

logger = init_logger()


def validate_prediction_quality(predictions_scaled, historical_data, stock_code):
    """
    验证预测质量，检查预测结果的合理性
    :param predictions_scaled: 预测价格数组
    :param historical_data: 历史数据
    :param stock_code: 股票代码
    :return: (is_valid, issues, suggestions)
    """
    issues = []
    suggestions = []
    
    if len(predictions_scaled) == 0:
        return False, ["预测结果为空"], ["检查模型训练是否成功"]
    
    # 1. 检查预测起始点连续性
    last_historical = historical_data['close'].iloc[-1]
    first_prediction = predictions_scaled[0]
    price_jump = abs(first_prediction - last_historical) / last_historical
    
    if price_jump > 0.05:  # 超过5%的跳跃
        issues.append(f"预测起始点跳跃过大：{price_jump:.2%}")
        suggestions.append("考虑调整预测起始点或重新训练模型")
    
    # 2. 检查预测价格范围（更严格的标准）
    historical_min = historical_data['close'].min()
    historical_max = historical_data['close'].max()
    historical_range = historical_max - historical_min
    
    pred_min = min(predictions_scaled)
    pred_max = max(predictions_scaled)
    
    # 检查是否超出历史价格范围过多（使用更保守的标准）
    if pred_min < historical_min * 0.85:  # 从0.8改为0.85
        issues.append(f"预测最低价过低：{pred_min:.2f} (历史最低: {historical_min:.2f})")
        suggestions.append("模型可能过度悲观，考虑调整训练参数或增加数据范围")
    
    if pred_max > historical_max * 1.15:  # 从1.2改为1.15
        issues.append(f"预测最高价过高：{pred_max:.2f} (历史最高: {historical_max:.2f})")
        suggestions.append("模型可能过度乐观，考虑调整训练参数")
    
    # 3. 检查预测波动性
    pred_changes = np.diff(predictions_scaled)
    pred_volatility = np.std(pred_changes)
    hist_volatility = historical_data['close'].pct_change().std()
    
    if pred_volatility < hist_volatility * 0.1:
        issues.append("预测波动性过低，可能过于平滑")
        suggestions.append("考虑增加模型对波动性的学习")
    
    if pred_volatility > hist_volatility * 3:
        issues.append("预测波动性过高，可能不稳定")
        suggestions.append("考虑减少噪声或调整模型参数")
    
    # 4. 检查预测趋势合理性
    pred_trend = (predictions_scaled[-1] - predictions_scaled[0]) / predictions_scaled[0]
    recent_trend = (historical_data['close'].iloc[-1] - historical_data['close'].iloc[-10]) / historical_data['close'].iloc[-10]
    
    # 如果预测趋势与近期趋势差异过大
    if abs(pred_trend - recent_trend) > 0.1:
        issues.append(f"预测趋势与近期趋势差异较大：预测{pred_trend:.2%} vs 近期{recent_trend:.2%}")
        suggestions.append("考虑增加近期数据权重或调整模型")
    
    # 5. 新增：检查预测值的分布合理性
    pred_mean = np.mean(predictions_scaled)
    hist_mean = historical_data['close'].mean()
    
    # 如果预测均值与历史均值差异过大
    mean_diff_ratio = abs(pred_mean - hist_mean) / hist_mean
    if mean_diff_ratio > 0.15:  # 超过15%的差异
        issues.append(f"预测均值与历史均值差异过大：{mean_diff_ratio:.2%}")
        suggestions.append("考虑检查训练数据的代表性或调整模型")
    
    # 6. 新增：检查预测的单调性
    # 如果预测呈现过于单调的趋势（连续上涨或下跌）
    monotonic_days = 0
    for i in range(1, len(predictions_scaled)):
        if (predictions_scaled[i] > predictions_scaled[i-1] and 
            predictions_scaled[i-1] > predictions_scaled[i-2] if i > 1 else True):
            monotonic_days += 1
        elif (predictions_scaled[i] < predictions_scaled[i-1] and 
              predictions_scaled[i-1] < predictions_scaled[i-2] if i > 1 else True):
            monotonic_days += 1
    
    if monotonic_days >= len(predictions_scaled) - 1:
        issues.append("预测趋势过于单调，缺乏必要的波动")
        suggestions.append("考虑增加随机性或调整模型参数")
    
    return len(issues) == 0, issues, suggestions


def predict_next_week(data_df, scaler, stock_code, model_path=LSTM_MODEL_PATH):
    """预测未来7天收盘价（文件名添加股票名称）"""
    # 1. 数据校验
    if data_df.empty:
        raise ValueError("历史数据为空，无法预测")
    missing_cols = [col for col in FEATURE_COLUMNS if col not in data_df.columns]
    if missing_cols:
        raise ValueError(f"数据缺少特征：{missing_cols}，需包含{FEATURE_COLUMNS}")

    # 2. 加载模型
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在：{model_path}")
        model = load_model(model_path)
        logger.info(f"成功加载模型：{model_path}")
    except Exception as e:
        raise ValueError(f"模型加载失败：{str(e)}") from e

    # 3. 准备输入数据（4个特征）
    data = data_df[FEATURE_COLUMNS].values
    scaled_data = scaler.transform(data)

    if len(scaled_data) < LOOK_BACK:
        raise ValueError(f"历史数据不足{LOOK_BACK}天，无法生成输入")
    input_data = scaled_data[-LOOK_BACK:]

    # 4. 预测未来7天
    predictions = []
    current_input = input_data.copy()
    
    # 获取最后一个历史收盘价作为基准
    last_historical_close = data_df['close'].iloc[-1]
    
    # 计算历史波动率
    historical_volatility = data_df['close'].pct_change().std()
    
    for day in range(PREDICTION_DAYS):
        # 重塑输入数据为模型期望的格式
        model_input = current_input.reshape(1, LOOK_BACK, len(FEATURE_COLUMNS))
        # 预测下一个时间步
        pred = model.predict(model_input, verbose=0)[0, 0]
        
        # 添加合理的随机波动（基于历史波动率）
        if day > 0:  # 第一天保持模型预测，后续添加波动
            # 生成基于历史波动率的随机变化
            random_change = np.random.normal(0, historical_volatility * 0.5)
            pred = pred * (1 + random_change)
        
        predictions.append(pred)
        
        # 更新输入数据（滑动窗口）
        # 创建新的输入行，用预测值填充第一个特征（收盘价），其他特征用历史数据的平均值
        new_row = np.zeros(len(FEATURE_COLUMNS))
        new_row[0] = pred  # 收盘价用预测值
        
        # 改进：其他特征使用更合理的值
        for i in range(1, len(FEATURE_COLUMNS)):
            if i == 1:  # high
                new_row[i] = max(pred, current_input[:, i].mean())  # 最高价不低于收盘价
            elif i == 2:  # low
                new_row[i] = min(pred, current_input[:, i].mean())  # 最低价不高于收盘价
            else:  # volume
                new_row[i] = current_input[:, i].mean()  # 成交量用平均值
        
        # 更新输入数据
        current_input = np.vstack([current_input[1:], new_row])

    # 5. 反归一化
    try:
        pred_array = np.array(predictions).reshape(-1, 1)
        full_pred = np.hstack([pred_array, np.zeros((len(pred_array), 3))])
        predictions_scaled = scaler.inverse_transform(full_pred)[:, 0]
        
        # 新增：调整预测起始点，确保与历史数据连续
        if len(predictions_scaled) > 0:
            # 计算历史价格与预测起始点的差异
            first_pred = predictions_scaled[0]
            price_diff = abs(first_pred - last_historical_close)
            
            # 如果差异过大（超过5%），进行平滑调整
            if price_diff / last_historical_close > 0.05:
                # 使用历史价格的趋势来调整预测起始点
                recent_trend = data_df['close'].tail(5).diff().mean()
                adjusted_first_pred = last_historical_close + recent_trend
                
                # 计算调整系数
                adjustment_factor = adjusted_first_pred / first_pred
                
                # 应用调整系数到所有预测值
                predictions_scaled = predictions_scaled * adjustment_factor
                
                logger.info(f"预测起始点调整：{first_pred:.2f} → {adjusted_first_pred:.2f} (调整系数: {adjustment_factor:.3f})")
        
        # 新增：预测范围控制
        historical_min = data_df['close'].min()
        historical_max = data_df['close'].max()
        historical_range = historical_max - historical_min
        
        # 设置合理的预测范围边界（基于历史数据的80%-120%）
        min_allowed = historical_min * 0.8
        max_allowed = historical_max * 1.2
        
        # 检查并调整超出范围的预测值
        adjusted_predictions = []
        for i, pred in enumerate(predictions_scaled):
            if pred < min_allowed:
                # 如果预测值过低，使用历史最低价的90%
                adjusted_pred = historical_min * 0.9
                logger.warning(f"第{i+1}天预测值过低({pred:.2f})，调整为{adjusted_pred:.2f}")
                adjusted_predictions.append(adjusted_pred)
            elif pred > max_allowed:
                # 如果预测值过高，使用历史最高价的110%
                adjusted_pred = historical_max * 1.1
                logger.warning(f"第{i+1}天预测值过高({pred:.2f})，调整为{adjusted_pred:.2f}")
                adjusted_predictions.append(adjusted_pred)
            else:
                adjusted_predictions.append(pred)
        
        predictions_scaled = np.array(adjusted_predictions)
        
        # 6. 新增：预测质量验证
        is_valid, issues, suggestions = validate_prediction_quality(predictions_scaled, data_df, stock_code)
        if not is_valid:
            logger.warning(f"预测质量检查发现问题：{issues}")
            logger.info(f"改进建议：{suggestions}")
        
        logger.info("未来7天预测完成，已反归一化并调整连续性")
    except Exception as e:
        raise ValueError(f"反归一化失败：{str(e)}") from e

    # 6. 生成未来日期（过滤非交易日）
    last_date = pd.to_datetime(data_df["date"].max())
    next_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
    # 过滤周末（周六和周日），得到有效的交易日期
    valid_next_dates = [d for d in next_dates if d.weekday() not in [5, 6]]
    # 如果过滤后不足7天，补充后续的日期直到满足7天（这里简单处理，可根据实际交易日历优化）
    while len(valid_next_dates) < PREDICTION_DAYS:
        next_candidate = valid_next_dates[-1] + timedelta(days=1)
        while next_candidate.weekday() in [5, 6]:
            next_candidate += timedelta(days=1)
        valid_next_dates.append(next_candidate)

    # 7. 整理预测结果
    pred_df = pd.DataFrame({
        "日期": [d.strftime("%Y-%m-%d") for d in valid_next_dates],
        "预测收盘价（元）": predictions_scaled[:len(valid_next_dates)].round(2)
    })
    
    # 控制台打印（新增股票名称显示）
    stock_name = get_stock_name_by_code(stock_code)  # 关键：获取股票名称
    print("\n" + "="*60)
    print(f"{stock_code}（{stock_name}）未来{PREDICTION_DAYS}天收盘价预测结果")  # 显示代码+名称
    print("="*60)
    print(pred_df.to_string(index=False))
    print("="*60 + "\n")
    
    # 8. 保存结果（文件名添加"代码+名称+日期"）
    create_dir(RESULT_DIR)
    # 关键修改：生成带名称的基础文件名
    code_suffix = stock_code.replace(".", "")  # sz.301551 → sz301551
    stock_name_safe = safe_filename(stock_name) if stock_name else ""
    current_date = datetime.now().strftime("%Y%m%d")
    # 最终基础文件名：sz301551_无线传媒_20250816
    base_filename = f"{code_suffix}_{stock_name_safe}_{current_date}" if stock_name_safe else f"{code_suffix}_{current_date}"

    # 8.1 保存Excel（带名称）
    excel_path = os.path.join(RESULT_DIR, f"{base_filename}_预测数据.xlsx")
    try:
        # 确保sheet_name不为None
        sheet_name = f"{stock_name}预测" if stock_name else "未来30天预测"
        pred_df.to_excel(excel_path, index=False, sheet_name=sheet_name, engine="openpyxl")
        logger.info(f"预测数据已保存至：{excel_path}")
    except Exception as e:
        raise ValueError(f"Excel保存失败：{str(e)}")

    # 8.2 绘制预测图（图表标题添加名称）
    fig_path = os.path.join(RESULT_DIR, f"{base_filename}_预测图表.png")
    try:
        plt.figure(figsize=CHART_FIGSIZE, dpi=CHART_DPI)
        # 历史数据（近30天）
        history_recent = data_df.sort_values("date").tail(30)
        plt.plot(pd.to_datetime(history_recent["date"]), history_recent["close"],
                 label="历史收盘价（近30天）", color="#1f77b4", linewidth=1.5)
        # 预测数据
        plt.plot(pd.to_datetime(pred_df["日期"]), pred_df["预测收盘价（元）"],
                 label="预测收盘价", color="#d62728", linewidth=2, linestyle="--")
        # 预测起点线
        plt.axvline(x=pd.to_datetime(data_df["date"].max()),
                    color="#7f7f7f", linestyle=":", linewidth=1.5, label="预测起始日")

        # 关键修改：图表标题显示"代码+名称"
        plt.title(f"{stock_code}（{stock_name}）未来{PREDICTION_DAYS}天收盘价预测", fontsize=14)
        plt.xlabel("日期", fontsize=12)
        plt.ylabel("价格（元）", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3, linestyle="--")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=CHART_DPI)
        plt.close()
        logger.info(f"预测图表已保存至：{fig_path}")
    except Exception as e:
        logger.warning(f"图表生成失败：{str(e)}")

    return pred_df


def get_prediction_summary(pred_df, stock_code, stock_name=""):
    """
    获取预测结果摘要
    :param pred_df: 预测结果数据框
    :param stock_code: 股票代码
    :param stock_name: 股票名称
    :return: 摘要信息字典
    """
    if pred_df.empty:
        return {"error": "预测结果为空"}
    
    prices = pred_df["预测收盘价（元）"].values
    price_changes = np.diff(prices)
    
    summary = {
        "股票代码": stock_code,
        "股票名称": stock_name,
        "预测天数": len(pred_df),
        "起始价格": round(prices[0], 2),
        "结束价格": round(prices[-1], 2),
        "价格变化": round(prices[-1] - prices[0], 2),
        "变化幅度": round(((prices[-1] - prices[0]) / prices[0]) * 100, 2),
        "最高价格": round(np.max(prices), 2),
        "最低价格": round(np.min(prices), 2),
        "平均价格": round(np.mean(prices), 2),
        "上涨天数": np.sum(price_changes > 0),
        "下跌天数": np.sum(price_changes < 0),
        "平盘天数": np.sum(price_changes == 0)
    }
    
    return summary


def validate_prediction_data(pred_df):
    """
    验证预测数据的合理性
    :param pred_df: 预测结果数据框
    :return: (is_valid, issues)
    """
    issues = []
    
    if pred_df.empty:
        issues.append("预测结果为空")
        return False, issues
    
    prices = pred_df["预测收盘价（元）"].values
    
    # 检查价格合理性
    if np.any(prices <= 0):
        issues.append("存在非正价格")
    
    # 检查价格变化幅度（单日变化不应超过20%）
    price_changes = np.abs(np.diff(prices) / prices[:-1]) * 100
    if np.any(price_changes > 20):
        issues.append("存在异常大的价格变化（>20%）")
    
    # 检查价格连续性
    if len(prices) < PREDICTION_DAYS:
        issues.append(f"预测天数不足：{len(prices)}天，期望{PREDICTION_DAYS}天")
    
    return len(issues) == 0, issues