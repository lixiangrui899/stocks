# 入口文件：启动GUI（保留手动输入+新增中文模糊搜索）
import tkinter as tk
import time
import os  # 新增：用于文件路径判断
import pandas as pd  # 新增：用于读取本地股票列表
from fuzzywuzzy import process  # 新增：用于中文模糊匹配
from tkinter import messagebox, ttk
import threading
from data_crawler import crawl_stock_data
from model_trainer import train_lstm_model
from predictor import predict_next_week
from utils import init_logger, create_dir, validate_stock_code, check_dependencies
from utils import get_stock_name_by_code  # 新增get_stock_name_by_code
from datetime import datetime  # 新增datetime（用于生成日期）
from config import (
    STOCK_LIST_PATH, RESULT_DIR, GUI_WIDTH, GUI_HEIGHT, GUI_TITLE,
    FUZZY_MATCH_THRESHOLD, SIMILAR_MATCH_THRESHOLD
)

logger = init_logger()
# 初始化时确保股票列表文件目录存在（避免首次使用报错）
create_dir(os.path.dirname(STOCK_LIST_PATH) if os.path.dirname(STOCK_LIST_PATH) else ".")


class StockPredictionApp:
    def __init__(self):
        self.root = None
        self.stock_code = None
        self.data_df = None
        self.scaler = None

    def run(self):
        """启动股票预测应用"""
        try:
            self.root = tk.Tk()
            self.root.title(GUI_TITLE)
            self.root.geometry(f"{GUI_WIDTH}x{GUI_HEIGHT}")  # 扩大窗口容纳中文搜索组件
            
            # 检查依赖
            self._check_dependencies()
            self._init_gui()
            
            # 启动主循环
            self.root.mainloop()
        except Exception as e:
            logger.critical(f"GUI启动失败：{str(e)}")
            messagebox.showerror("启动失败", f"程序启动异常：{str(e)}\n建议：1. 检查依赖 2. 以管理员身份运行")

    def _check_dependencies(self):
        """检查必要的依赖包"""
        is_ready, missing_packages = check_dependencies()
        if not is_ready:
            error_msg = f"缺少必要的依赖包：{', '.join(missing_packages)}\n请运行：pip install {' '.join(missing_packages)}"
            messagebox.showerror("依赖缺失", error_msg)
            logger.error(f"依赖包缺失：{missing_packages}")

    def _init_gui(self):
        """初始化GUI（新增中文搜索组件，保留手动输入）"""
        # 标题栏（更新标题，包含中文搜索功能说明）
        title_label = ttk.Label(
            self.root,
            text="股票预测系统（中文搜索/手动输入代码）",
            font=("微软雅黑", 12, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

        # 新增：1. 中文名称搜索区域（核心新增功能）
        ttk.Label(self.root, text="中文名称搜索（推荐）：").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.company_entry = ttk.Entry(self.root, width=30)  # 中文名称输入框
        self.company_entry.grid(row=1, column=1, padx=10, pady=10)
        self.search_btn = ttk.Button(
            self.root,
            text="搜索股票代码",
            command=self._search_code_local,  # 绑定模糊搜索方法
            style="Normal.TButton"
        )
        self.search_btn.grid(row=1, column=2, padx=10, pady=10)
        # 中文搜索提示（帮助用户理解）
        ttk.Label(
            self.root,
            text="示例：输入\"无线传媒\"可匹配\"sz.301551\"",
            foreground="gray",
            font=("微软雅黑", 8)
        ).grid(row=2, column=1, padx=10, pady=0, sticky="w")

        # 保留：2. 手动输入代码区域
        ttk.Label(self.root, text="手动输入代码（备用）：").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.manual_code_entry = ttk.Entry(self.root, width=30)
        self.manual_code_entry.grid(row=3, column=1, padx=10, pady=10)
        ttk.Label(
            self.root,
            text="格式示例：sz.000858（五粮液）/sh.600519（茅台）",
            foreground="gray",
            font=("微软雅黑", 8)
        ).grid(row=4, column=1, padx=10, pady=0, sticky="w")
        self.manual_btn = ttk.Button(self.root, text="确认股票代码", command=self._use_manual_code)
        self.manual_btn.grid(row=5, column=1, padx=10, pady=5, sticky="e")

        # 保留：3. 代码确认结果
        ttk.Label(self.root, text="当前代码：").grid(row=6, column=0, padx=10, pady=10, sticky="w")
        self.code_var = tk.StringVar(value="未确认（请搜索或手动输入代码）")
        self.code_label = ttk.Label(self.root, textvariable=self.code_var, foreground="blue", font=("微软雅黑", 10))
        self.code_label.grid(row=6, column=1, padx=10, pady=10, sticky="w")

        # 保留：4. 开始按钮
        self.run_btn = ttk.Button(
            self.root,
            text="开始：爬取数据→训练模型→预测",
            command=self._start_full_process,
            state="disabled",
            style="Accent.TButton"
        )
        self.run_btn.grid(row=7, column=1, padx=10, pady=10)

        # 保留：5. 进度显示
        self.progress_var = tk.StringVar(value="等待开始...")
        progress_label = ttk.Label(self.root, textvariable=self.progress_var, foreground="green", font=("微软雅黑", 10))
        progress_label.grid(row=8, column=0, columnspan=3, padx=10, pady=10)

        # 设置按钮样式
        style = ttk.Style()
        style.configure("Accent.TButton", font=("微软雅黑", 10, "bold"))

    def _search_code_local(self):
        """新增：本地中文模糊搜索（核心功能）"""
        company_name = self.company_entry.get().strip()
        if not company_name:
            messagebox.showwarning("警告", "请输入公司名称")
            return

        try:
            # 读取本地股票列表
            if not os.path.exists(STOCK_LIST_PATH):
                raise FileNotFoundError(f"股票列表文件不存在：{STOCK_LIST_PATH}")

            stock_df = pd.read_csv(STOCK_LIST_PATH, encoding='utf-8')
            if stock_df.empty:
                raise ValueError("股票列表文件为空")

            # 提取公司名称列表（用于模糊匹配）
            company_names = stock_df['name'].tolist()
            
            # 使用fuzzywuzzy进行模糊匹配
            matches = process.extract(company_name, company_names, limit=5)
            
            # 找到最佳匹配
            best_match = None
            best_score = 0
            
            for match_name, match_score in matches:
                if match_score >= FUZZY_MATCH_THRESHOLD:  # 使用配置的阈值
                    # 获取对应的股票代码
                    matched_row = stock_df[stock_df['name'] == match_name].iloc[0]
                    matched_code = matched_row['code']
                    matched_name = matched_row['name']
                    
                    if match_score > best_score:
                        best_match = (matched_code, matched_name, match_score)
                        best_score = match_score
            
            if best_match:
                matched_code, matched_name, match_score = best_match
                self.stock_code = matched_code
                self.code_var.set(f"✅ 匹配成功：{company_name} → {matched_code}（{matched_name}）")
                self.run_btn.config(state="normal")
                # 清空手动输入框（避免用户混淆）
                self.manual_code_entry.delete(0, tk.END)
                logger.info(f"中文搜索匹配成功：{company_name} → {matched_code}（匹配度：{match_score}%）")
            else:
                # 如果没有找到高匹配度的结果，显示所有可能的匹配
                suggestions = [f"{name}（{stock_df[stock_df['name'] == name].iloc[0]['code']}）" 
                             for name, score in matches if score >= SIMILAR_MATCH_THRESHOLD]
                
                if suggestions:
                    suggestion_text = "\n".join(suggestions[:3])  # 显示前3个建议
                    messagebox.showinfo(
                        "未找到精确匹配", 
                        f"未找到与\"{company_name}\"精确匹配的公司\n\n可能的匹配：\n{suggestion_text}\n\n请尝试更精确的公司名称"
                    )
                else:
                    messagebox.showwarning("未找到匹配", f"未找到与\"{company_name}\"相关的公司\n请检查公司名称是否正确")

        except Exception as e:
            # 捕获所有异常，友好提示用户
            error_msg = str(e)
            self.code_var.set(f"❌ 搜索失败：{error_msg[:25]}...")
            logger.warning(f"中文搜索失败：{error_msg}")
            messagebox.showerror("搜索失败", f"错误原因：{error_msg}\n建议：检查stock_list.csv文件是否正常")

    def _use_manual_code(self):
        """保留：手动输入代码校验逻辑（使用新的验证函数）"""
        code = self.manual_code_entry.get().strip()
        is_valid, error_msg = validate_stock_code(code)
        if not is_valid:
            messagebox.showerror("错误", error_msg)
            return

        self.stock_code = code
        self.code_var.set(f"✅ 已确认：{code}")
        self.run_btn.config(state="normal")
        # 清空搜索框（避免用户混淆）
        self.company_entry.delete(0, tk.END)

    def _start_full_process(self):
        """保留：启动全流程逻辑（无修改）"""
        self.run_btn.config(state="disabled")
        self.manual_btn.config(state="disabled")
        self.search_btn.config(state="disabled")  # 新增：搜索按钮也禁用，避免重复操作
        self.progress_var.set("🚀 流程启动中...（请勿关闭窗口，约3-8分钟）")

        threading.Thread(target=self._full_process, daemon=True).start()

    def _full_process(self):
        """保留：全流程逻辑（无修改）"""
        try:
            # 1. 爬取数据
            self.progress_var.set("1/3：正在爬取历史数据...")
            self.data_df, csv_path = crawl_stock_data(self.stock_code, days=6000)
            data_days = len(self.data_df)
            self.progress_var.set(f"1/3：数据爬取完成（共{data_days}天有效数据）")
            logger.info(f"爬取到{data_days}天有效数据")
            time.sleep(1)

            # 2. 训练模型
            self.progress_var.set("2/3：正在训练LSTM模型...（约2-5分钟）")
            model_path, self.scaler = train_lstm_model(self.data_df)

            # 3. 预测未来7天
            self.progress_var.set("3/3：正在预测未来7天数据...（约30秒）")
            pred_df = predict_next_week(
                self.data_df,
                self.scaler,
                stock_code=self.stock_code,
                model_path="models/lstm_stock_model.keras"
            )

            stock_name = get_stock_name_by_code(self.stock_code)  # 调用工具函数获取名称
            self.progress_var.set("🎉 全流程完成！结果已保存至对应文件夹")
            messagebox.showinfo(
                "操作成功",
                f"✅ 股票信息：{self.stock_code}（{stock_name}）\n"  # 显示代码+名称
                f"✅ 历史数据：{csv_path}\n"
                f"✅ 训练模型：{model_path}\n"
                f"✅ 预测数据：{RESULT_DIR}{self.stock_code.replace('.', '')}_{stock_name.replace('/', '_') if stock_name else ''}_{datetime.now().strftime('%Y%m%d')}_预测数据.xlsx\n"  # 带名称的路径
                f"✅ 预测图表：{RESULT_DIR}{self.stock_code.replace('.', '')}_{stock_name.replace('/', '_') if stock_name else ''}_{datetime.now().strftime('%Y%m%d')}_预测图表.png\n"
                f"✅ 训练损失曲线：figures/lstm_training_loss.png"
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"全流程失败：{error_msg}")
            if "爬取" in error_msg or "网络" in error_msg:
                error_msg += "\n建议：1. 检查网络 2. 确认代码有效 3. 稍后重试"
            elif "训练" in error_msg or "模型" in error_msg:
                error_msg += "\n建议：1. 确保数据非空 2. 检查TensorFlow环境"
            elif "预测" in error_msg:
                error_msg += "\n建议：1. 确保模型训练成功 2. 检查数据格式"

            self.progress_var.set(f"❌ 流程失败：{error_msg[:25]}...")
            messagebox.showerror("操作失败", error_msg)

        finally:
            # 解锁所有按钮（包括新增的搜索按钮）
            self.run_btn.config(state="normal")
            self.manual_btn.config(state="normal")
            self.search_btn.config(state="normal")


# 保持向后兼容性
class StockPredictGUI(StockPredictionApp):
    """向后兼容的类名"""
    pass


if __name__ == "__main__":
    try:
        app = StockPredictionApp()
        app.run()
    except Exception as global_e:
        logger.critical(f"GUI启动失败：{str(global_e)}")
        messagebox.showerror("启动失败", f"程序启动异常：{str(global_e)}\n建议：1. 检查依赖 2. 以管理员身份运行")