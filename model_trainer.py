import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from utils import create_dir, init_logger
from config import (
    FIGURE_DIR, MODEL_DIR, FEATURE_COLUMNS, LOOK_BACK, 
    EPOCHS, BATCH_SIZE, LSTM_MODEL_PATH, BEST_MODEL_PATH,
    CHART_DPI, CHART_FIGSIZE
)

logger = init_logger()


def train_lstm_model(data_df, look_back=LOOK_BACK, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    训练LSTM模型（根治图表异常的核心修改）
    1. 强制复权校验：确保训练数据是后复权的
    2. 特征绑定校验：训练/预测特征100%一致
    3. 趋势学习增强：让模型必须学习特征间的联动关系
    4. 可视化强制对齐：训练曲线直接反映预测能力
    """
    # ======================== 1. 数据强校验（解决历史曲线异常） ========================
    # 1.1 复权校验（必须后复权，否则历史曲线断层）
    if 'adjustflag' not in data_df.columns or data_df['adjustflag'].max() != '3':
        raise ValueError(
            "训练数据未做后复权！请在 data_crawler.py 中设置 adjustflag='3'，否则历史曲线会异常"
        )

    # 1.2 特征严格对齐（与预测端完全一致）
    missing_cols = [col for col in FEATURE_COLUMNS if col not in data_df.columns]
    if missing_cols:
        raise ValueError(
            f"训练特征与预测特征不一致！预测需 {FEATURE_COLUMNS}，但数据缺少 {missing_cols}"
        )

    # ======================== 2. 数据预处理（让模型学懂趋势） ========================
    # 2.1 多特征归一化（保留维度，让模型学联动关系）
    features = data_df[FEATURE_COLUMNS].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(features)

    # 2.2 数据增强：添加噪声和波动性
    def add_noise_to_data(data, noise_factor=0.01):
        """为训练数据添加少量噪声，提高模型鲁棒性"""
        noise = np.random.normal(0, noise_factor, data.shape)
        return data + noise
    
    # 对训练数据进行轻微增强
    scaled_data_augmented = add_noise_to_data(scaled_data, noise_factor=0.005)

    # 2.3 构建时序数据集（强制让模型看够历史趋势）
    X, y = [], []
    for i in range(look_back, len(scaled_data_augmented)):
        # 让模型必须学习多特征的时间关联（如收盘价与成交量的关系）
        X.append(scaled_data_augmented[i - look_back:i, :])
        y.append(scaled_data_augmented[i, 0])  # 只预测收盘价（第0列）

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    logger.info(
        f"训练数据构建完成：{len(X)}条样本，"
        f"特征维度 {X.shape[1]}天×{X.shape[2]}特征（与预测严格对齐）"
    )

    # 2.4 数据集划分（8:2，确保验证集能反映预测能力）
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # ======================== 3. 模型重构（让预测曲线贴合趋势） ========================
    model = Sequential([
        # 第一层：必须保留序列输出，让模型学多特征的时间联动
        LSTM(
            units=64,
            return_sequences=True,
            input_shape=(look_back, len(FEATURE_COLUMNS)),
            dropout=0.2  # 内置dropout，比单独层更高效
        ),

        # 第二层：压缩维度，聚焦关键趋势
        LSTM(
            units=32,
            return_sequences=False,
            dropout=0.1
        ),

        # 回归头：直接映射到预测值（拒绝多余激活函数干扰）
        Dense(units=1)
    ])

    # 3.2 编译优化（用学习率调度器，让模型前期快学、后期精调）
    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=0.001, amsgrad=True),  # 优化器增强
        loss="mean_squared_error",
        metrics=["mae", "mape"]  # 增加MAPE，直接看预测误差率
    )

    # ======================== 4. 训练增强（解决预测趋势不合理） ========================
    # 4.1 早停+checkpoint（保存最会预测的模型，不是最会拟合的）
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    early_stop = EarlyStopping(
        monitor="val_mape",  # 用误差率监控，比loss更直观
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor="val_mape",
        save_best_only=True,
        verbose=1
    )

    # 4.2 学习率调度（避免模型卡在局部最优）
    lr_scheduler = ReduceLROnPlateau(
        monitor="val_mape",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    # 4.3 启动训练（让验证集直接反映预测能力）
    logger.info(f"开始训练（目标：让验证集MAPE<5%，epochs={epochs}）")
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, checkpoint, lr_scheduler],
        verbose=1
    )

    # ======================== 5. 可视化重构（直接指导预测优化） ========================
    try:
        create_dir(FIGURE_DIR)
        plt.figure(figsize=CHART_FIGSIZE, dpi=CHART_DPI)

        # 5.1 损失曲线（MSE+MAPE双维度）
        plt.subplot(2, 1, 1)
        plt.plot(history.history["loss"], label="训练MSE", color="#2ca02c")
        plt.plot(history.history["val_loss"], label="验证MSE", color="#d62728")
        plt.title("训练损失（MSE）", fontsize=12)
        plt.ylabel("损失值", fontsize=10)
        plt.legend(fontsize=9)
        plt.grid(alpha=0.3)

        # 5.2 误差率曲线（MAPE更直观）
        plt.subplot(2, 1, 2)
        plt.plot(history.history["mape"], label="训练MAPE", color="#1f77b4")
        plt.plot(history.history["val_mape"], label="验证MAPE", color="#ff7f0e")
        plt.title("预测误差率（MAPE）", fontsize=12)
        plt.xlabel("训练轮次", fontsize=10)
        plt.ylabel("误差率%", fontsize=10)
        plt.legend(fontsize=9)
        plt.grid(alpha=0.3)

        # 5.3 强制用Agg，兼容所有环境
        import matplotlib
        matplotlib.use('Agg')
        loss_fig_path = os.path.join(FIGURE_DIR, "lstm_training_metrics.png")
        plt.savefig(loss_fig_path, bbox_inches="tight", dpi=CHART_DPI)
        plt.close()
        logger.info(f"训练指标已保存至：{loss_fig_path}（看MAPE判断预测能力）")

    except Exception as e:
        logger.error(f"可视化失败：{str(e)}，但模型已保存")

    # ======================== 6. 模型保存（只留最会预测的） ========================
    # 6.1 删除无效模型（避免混淆）
    if os.path.exists(LSTM_MODEL_PATH):
        os.remove(LSTM_MODEL_PATH)
    # 6.2 重命名最佳模型
    import shutil
    shutil.copy(BEST_MODEL_PATH, LSTM_MODEL_PATH)
    logger.info(f"最佳预测模型已保存至：{LSTM_MODEL_PATH}（验证集MAPE={min(history.history['val_mape']):.2f}%）")

    return LSTM_MODEL_PATH, scaler


def evaluate_model_performance(model, X_test, y_test, scaler):
    """
    评估模型性能
    :param model: 训练好的模型
    :param X_test: 测试数据
    :param y_test: 测试标签
    :param scaler: 数据缩放器
    :return: 性能指标字典
    """
    # 预测
    y_pred = model.predict(X_test, verbose=0)
    
    # 反归一化
    y_test_original = scaler.inverse_transform(
        np.hstack([y_test.reshape(-1, 1), np.zeros((len(y_test), 3))])
    )[:, 0]
    y_pred_original = scaler.inverse_transform(
        np.hstack([y_pred, np.zeros((len(y_pred), 3))])
    )[:, 0]
    
    # 计算指标
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
    
    mse = mean_squared_error(y_test_original, y_pred_original)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    mape = mean_absolute_percentage_error(y_test_original, y_pred_original) * 100
    
    # 计算方向准确率（预测涨跌的准确性）
    direction_accuracy = np.mean(
        np.sign(np.diff(y_test_original)) == np.sign(np.diff(y_pred_original))
    ) * 100
    
    return {
        "MSE": round(mse, 4),
        "MAE": round(mae, 4),
        "MAPE": round(mape, 2),
        "方向准确率": round(direction_accuracy, 2)
    }


def get_model_summary(model):
    """
    获取模型摘要信息
    :param model: Keras模型
    :return: 模型摘要信息
    """
    summary = []
    model.summary(print_fn=lambda x: summary.append(x))
    return "\n".join(summary)