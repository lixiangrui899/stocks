import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import subprocess
import sys
from datetime import datetime
import threading

# 导入项目模块
from main import StockPredictionApp
from config import RESULT_DIR, FIGURE_DIR
from utils import init_logger, safe_filename

logger = init_logger()

class MainGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("股票预测系统 - 主界面")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # 设置样式
        style = ttk.Style()
        style.theme_use('clam')
        
        # 创建主框架
        self.main_frame = ttk.Frame(root, padding="20")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        
        # 创建标题
        title_label = ttk.Label(self.main_frame, text="股票预测系统", 
                               font=("微软雅黑", 24, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 30))
        
        # 创建功能按钮
        self.create_buttons()
        
        # 创建状态显示区域
        self.create_status_area()
        
        # 创建日志显示区域
        self.create_log_area()
        
        # 初始化股票预测应用
        self.stock_app = None
        
    def create_buttons(self):
        """创建功能按钮"""
        # 按钮框架
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=1, column=0, columnspan=3, pady=(0, 20))
        
        # 主要功能按钮
        self.btn_stock_prediction = ttk.Button(
            button_frame, 
            text="股票预测", 
            command=self.open_stock_prediction,
            style="Accent.TButton"
        )
        self.btn_stock_prediction.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.btn_view_charts = ttk.Button(
            button_frame, 
            text="查看历史图表", 
            command=self.open_chart_viewer
        )
        self.btn_view_charts.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.btn_view_data = ttk.Button(
            button_frame, 
            text="查看历史数据", 
            command=self.open_data_viewer
        )
        self.btn_view_data.pack(side=tk.LEFT, padx=10, pady=10)
        
        # 设置按钮样式
        style = ttk.Style()
        style.configure("Accent.TButton", font=("微软雅黑", 12, "bold"))
        
    def create_status_area(self):
        """创建状态显示区域"""
        status_frame = ttk.LabelFrame(self.main_frame, text="系统状态", padding="10")
        status_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        status_frame.columnconfigure(0, weight=1)
        
        # 状态信息
        self.status_text = tk.StringVar()
        self.status_text.set("系统就绪")
        status_label = ttk.Label(status_frame, textvariable=self.status_text)
        status_label.grid(row=0, column=0, sticky=tk.W)
        
        # 进度条
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
    def create_log_area(self):
        """创建日志显示区域"""
        log_frame = ttk.LabelFrame(self.main_frame, text="系统日志", padding="10")
        log_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # 日志文本框
        self.log_text = tk.Text(log_frame, height=15, width=80, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # 清空日志按钮
        clear_btn = ttk.Button(log_frame, text="清空日志", command=self.clear_log)
        clear_btn.grid(row=1, column=0, pady=(5, 0))
        
    def open_stock_prediction(self):
        """打开股票预测界面"""
        try:
            self.log_message("正在启动股票预测功能...")
            self.status_text.set("启动股票预测界面")
            self.progress.start()
            
            # 在新线程中启动股票预测应用
            def run_prediction():
                try:
                    if self.stock_app is None:
                        self.stock_app = StockPredictionApp()
                    self.stock_app.run()
                    self.root.after(0, lambda: self.log_message("股票预测功能已启动"))
                except Exception as e:
                    self.root.after(0, lambda: self.log_message(f"启动股票预测失败: {str(e)}"))
                finally:
                    self.root.after(0, lambda: self.progress.stop())
                    self.root.after(0, lambda: self.status_text.set("系统就绪"))
            
            thread = threading.Thread(target=run_prediction, daemon=True)
            thread.start()
            
        except Exception as e:
            self.log_message(f"启动股票预测失败: {str(e)}")
            self.progress.stop()
            self.status_text.set("系统就绪")
    
    def open_chart_viewer(self):
        """打开图表查看器"""
        try:
            self.log_message("正在打开图表查看器...")
            ChartViewer(self.root, self)
        except Exception as e:
            self.log_message(f"打开图表查看器失败: {str(e)}")
    
    def open_data_viewer(self):
        """打开数据查看器"""
        try:
            self.log_message("正在打开数据查看器...")
            DataViewer(self.root, self)
        except Exception as e:
            self.log_message(f"打开数据查看器失败: {str(e)}")
    
    def log_message(self, message):
        """添加日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        
        # 限制日志行数
        lines = self.log_text.get("1.0", tk.END).split('\n')
        if len(lines) > 100:
            self.log_text.delete("1.0", f"{len(lines)-100}.0")
    
    def clear_log(self):
        """清空日志"""
        self.log_text.delete("1.0", tk.END)


class ChartViewer:
    def __init__(self, parent, main_gui):
        self.parent = parent
        self.main_gui = main_gui
        self.window = tk.Toplevel(parent)
        self.window.title("历史图表查看器")
        self.window.geometry("900x700")
        self.window.resizable(True, True)
        
        # 创建界面
        self.create_widgets()
        self.load_charts()
        
    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = ttk.Label(main_frame, text="历史图表查看器", 
                               font=("微软雅黑", 18, "bold"))
        title_label.pack(pady=(0, 20))
        
        # 搜索框架
        search_frame = ttk.Frame(main_frame)
        search_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(search_frame, text="搜索:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=30)
        self.search_entry.pack(side=tk.LEFT, padx=(5, 10))
        self.search_var.trace('w', self.filter_charts)
        
        # 刷新按钮
        refresh_btn = ttk.Button(search_frame, text="刷新", command=self.load_charts)
        refresh_btn.pack(side=tk.LEFT, padx=5)
        
        # 图表列表框架
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 创建Treeview
        columns = ("文件名", "股票代码", "股票名称", "生成日期", "文件大小")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=15)
        
        # 设置列标题
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 绑定双击事件
        self.tree.bind("<Double-1>", self.open_chart)
        
        # 按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="打开选中图表", command=self.open_selected_chart).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="在文件夹中显示", command=self.show_in_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="关闭", command=self.window.destroy).pack(side=tk.RIGHT, padx=5)
        
    def load_charts(self):
        """加载图表列表"""
        try:
            # 清空现有项目
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # 获取图表文件
            chart_files = []
            if os.path.exists(RESULT_DIR):
                for file in os.listdir(RESULT_DIR):
                    if file.endswith('.png') and '预测图表' in file:
                        file_path = os.path.join(RESULT_DIR, file)
                        chart_files.append(file_path)
            
            # 解析文件名并添加到列表
            for file_path in chart_files:
                filename = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)
                
                # 解析文件名格式: sh600519_贵州茅台_20250816_预测图表.png
                parts = filename.replace('_预测图表.png', '').split('_')
                if len(parts) >= 3:
                    stock_code = parts[0]
                    stock_name = parts[1]
                    date_str = parts[2]
                else:
                    stock_code = "未知"
                    stock_name = "未知"
                    date_str = "未知"
                
                # 格式化文件大小
                size_str = self.format_file_size(file_size)
                
                self.tree.insert("", tk.END, values=(filename, stock_code, stock_name, date_str, size_str))
            
            self.main_gui.log_message(f"加载了 {len(chart_files)} 个图表文件")
            
        except Exception as e:
            self.main_gui.log_message(f"加载图表列表失败: {str(e)}")
    
    def filter_charts(self, *args):
        """过滤图表列表"""
        search_term = self.search_var.get().lower()
        
        # 隐藏所有项目
        for item in self.tree.get_children():
            self.tree.detach(item)
        
        # 重新显示匹配的项目
        for item in self.tree.get_children():
            values = self.tree.item(item, "values")
            if any(search_term in str(value).lower() for value in values):
                self.tree.reattach(item, "", "end")
    
    def open_selected_chart(self):
        """打开选中的图表"""
        selection = self.tree.selection()
        if selection:
            item = selection[0]
            filename = self.tree.item(item, "values")[0]
            file_path = os.path.join(RESULT_DIR, filename)
            self.open_chart_file(file_path)
        else:
            messagebox.showwarning("警告", "请先选择一个图表文件")
    
    def open_chart(self, event):
        """双击打开图表"""
        item = self.tree.selection()[0]
        filename = self.tree.item(item, "values")[0]
        file_path = os.path.join(RESULT_DIR, filename)
        self.open_chart_file(file_path)
    
    def open_chart_file(self, file_path):
        """打开图表文件"""
        try:
            if os.path.exists(file_path):
                if sys.platform == "win32":
                    os.startfile(file_path)
                elif sys.platform == "darwin":  # macOS
                    subprocess.run(["open", file_path])
                else:  # Linux
                    subprocess.run(["xdg-open", file_path])
                self.main_gui.log_message(f"已打开图表: {os.path.basename(file_path)}")
            else:
                # 尝试使用绝对路径
                abs_path = os.path.abspath(file_path)
                if os.path.exists(abs_path):
                    if sys.platform == "win32":
                        os.startfile(abs_path)
                    elif sys.platform == "darwin":  # macOS
                        subprocess.run(["open", abs_path])
                    else:  # Linux
                        subprocess.run(["xdg-open", abs_path])
                    self.main_gui.log_message(f"已打开图表: {os.path.basename(abs_path)}")
                else:
                    messagebox.showerror("错误", f"文件不存在: {file_path}\n绝对路径: {abs_path}")
        except Exception as e:
            messagebox.showerror("错误", f"无法打开文件: {str(e)}")
    
    def show_in_folder(self):
        """在文件夹中显示文件"""
        selection = self.tree.selection()
        if selection:
            item = selection[0]
            filename = self.tree.item(item, "values")[0]
            file_path = os.path.join(RESULT_DIR, filename)
            
            try:
                if sys.platform == "win32":
                    subprocess.run(["explorer", "/select,", file_path])
                elif sys.platform == "darwin":  # macOS
                    subprocess.run(["open", "-R", file_path])
                else:  # Linux
                    subprocess.run(["xdg-open", os.path.dirname(file_path)])
                self.main_gui.log_message(f"已在文件夹中显示: {filename}")
            except Exception as e:
                messagebox.showerror("错误", f"无法打开文件夹: {str(e)}")
        else:
            messagebox.showwarning("警告", "请先选择一个文件")
    
    def format_file_size(self, size_bytes):
        """格式化文件大小"""
        if size_bytes == 0:
            return "0 B"
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        return f"{size_bytes:.1f} {size_names[i]}"


class DataViewer:
    def __init__(self, parent, main_gui):
        self.parent = parent
        self.main_gui = main_gui
        self.window = tk.Toplevel(parent)
        self.window.title("历史数据查看器")
        self.window.geometry("900x700")
        self.window.resizable(True, True)
        
        # 创建界面
        self.create_widgets()
        self.load_data_files()
        
    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = ttk.Label(main_frame, text="历史数据查看器", 
                               font=("微软雅黑", 18, "bold"))
        title_label.pack(pady=(0, 20))
        
        # 搜索框架
        search_frame = ttk.Frame(main_frame)
        search_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(search_frame, text="搜索:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=30)
        self.search_entry.pack(side=tk.LEFT, padx=(5, 10))
        self.search_var.trace('w', self.filter_data_files)
        
        # 刷新按钮
        refresh_btn = ttk.Button(search_frame, text="刷新", command=self.load_data_files)
        refresh_btn.pack(side=tk.LEFT, padx=5)
        
        # 数据文件列表框架
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 创建Treeview
        columns = ("文件名", "股票代码", "股票名称", "生成日期", "文件大小")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=15)
        
        # 设置列标题
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 绑定双击事件
        self.tree.bind("<Double-1>", self.open_data_file)
        
        # 按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="打开选中文件", command=self.open_selected_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="在文件夹中显示", command=self.show_in_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="关闭", command=self.window.destroy).pack(side=tk.RIGHT, padx=5)
        
    def load_data_files(self):
        """加载数据文件列表"""
        try:
            # 清空现有项目
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # 获取数据文件
            data_files = []
            if os.path.exists(RESULT_DIR):
                for file in os.listdir(RESULT_DIR):
                    if file.endswith('.xlsx') and '预测数据' in file:
                        file_path = os.path.join(RESULT_DIR, file)
                        data_files.append(file_path)
            
            # 解析文件名并添加到列表
            for file_path in data_files:
                filename = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)
                
                # 解析文件名格式: sh600519_贵州茅台_20250816_预测数据.xlsx
                parts = filename.replace('_预测数据.xlsx', '').split('_')
                if len(parts) >= 3:
                    stock_code = parts[0]
                    stock_name = parts[1]
                    date_str = parts[2]
                else:
                    stock_code = "未知"
                    stock_name = "未知"
                    date_str = "未知"
                
                # 格式化文件大小
                size_str = self.format_file_size(file_size)
                
                self.tree.insert("", tk.END, values=(filename, stock_code, stock_name, date_str, size_str))
            
            self.main_gui.log_message(f"加载了 {len(data_files)} 个数据文件")
            
        except Exception as e:
            self.main_gui.log_message(f"加载数据文件列表失败: {str(e)}")
    
    def filter_data_files(self, *args):
        """过滤数据文件列表"""
        search_term = self.search_var.get().lower()
        
        # 隐藏所有项目
        for item in self.tree.get_children():
            self.tree.detach(item)
        
        # 重新显示匹配的项目
        for item in self.tree.get_children():
            values = self.tree.item(item, "values")
            if any(search_term in str(value).lower() for value in values):
                self.tree.reattach(item, "", "end")
    
    def open_selected_file(self):
        """打开选中的文件"""
        selection = self.tree.selection()
        if selection:
            item = selection[0]
            filename = self.tree.item(item, "values")[0]
            file_path = os.path.join(RESULT_DIR, filename)
            self.open_data_file(file_path)
        else:
            messagebox.showwarning("警告", "请先选择一个数据文件")
    
    def open_data_file(self, event=None):
        """双击打开数据文件"""
        if event:
            item = self.tree.selection()[0]
        else:
            selection = self.tree.selection()
            if not selection:
                return
            item = selection[0]
        
        filename = self.tree.item(item, "values")[0]
        file_path = os.path.join(RESULT_DIR, filename)
        
        try:
            if os.path.exists(file_path):
                if sys.platform == "win32":
                    os.startfile(file_path)
                elif sys.platform == "darwin":  # macOS
                    subprocess.run(["open", file_path])
                else:  # Linux
                    subprocess.run(["xdg-open", file_path])
                self.main_gui.log_message(f"已打开数据文件: {filename}")
            else:
                # 尝试使用绝对路径
                abs_path = os.path.abspath(file_path)
                if os.path.exists(abs_path):
                    if sys.platform == "win32":
                        os.startfile(abs_path)
                    elif sys.platform == "darwin":  # macOS
                        subprocess.run(["open", abs_path])
                    else:  # Linux
                        subprocess.run(["xdg-open", abs_path])
                    self.main_gui.log_message(f"已打开数据文件: {filename}")
                else:
                    messagebox.showerror("错误", f"文件不存在: {file_path}\n绝对路径: {abs_path}")
        except Exception as e:
            messagebox.showerror("错误", f"无法打开文件: {str(e)}")
    
    def show_in_folder(self):
        """在文件夹中显示文件"""
        selection = self.tree.selection()
        if selection:
            item = selection[0]
            filename = self.tree.item(item, "values")[0]
            file_path = os.path.join(RESULT_DIR, filename)
            
            try:
                if sys.platform == "win32":
                    subprocess.run(["explorer", "/select,", file_path])
                elif sys.platform == "darwin":  # macOS
                    subprocess.run(["open", "-R", file_path])
                else:  # Linux
                    subprocess.run(["xdg-open", os.path.dirname(file_path)])
                self.main_gui.log_message(f"已在文件夹中显示: {filename}")
            except Exception as e:
                messagebox.showerror("错误", f"无法打开文件夹: {str(e)}")
        else:
            messagebox.showwarning("警告", "请先选择一个文件")
    
    def format_file_size(self, size_bytes):
        """格式化文件大小"""
        if size_bytes == 0:
            return "0 B"
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        return f"{size_bytes:.1f} {size_names[i]}"


def main():
    """主函数"""
    try:
        # 初始化必要的目录
        from config import init_directories
        init_directories()
        
        # 设置工作目录为exe所在目录
        if getattr(sys, 'frozen', False):
            # 如果是exe运行
            application_path = os.path.dirname(sys.executable)
            os.chdir(application_path)
            print(f"设置工作目录为: {application_path}")
        
        root = tk.Tk()
        app = MainGUI(root)
        
        # 设置窗口关闭事件
        def on_closing():
            if messagebox.askokcancel("退出", "确定要退出程序吗？"):
                root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # 启动主循环
        root.mainloop()
        
    except Exception as e:
        # 显示错误信息
        error_msg = f"程序启动失败: {str(e)}"
        print(error_msg)
        
        # 尝试显示错误对话框
        try:
            import tkinter.messagebox as messagebox
            messagebox.showerror("启动错误", error_msg)
        except:
            pass
        
        # 等待用户按键
        input("按回车键退出...")


if __name__ == "__main__":
    main()
