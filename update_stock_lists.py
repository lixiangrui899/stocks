import os
import baostock as bs
import pandas as pd
from datetime import datetime, timedelta
from utils import init_logger, create_dir

logger = init_logger()

STOCK_LIST_PATH = "stock_list.csv"
VALID_MARKETS = ["sh", "sz"]
VALID_CODE_PREFIX = ["00", "60", "30", "68"]

def get_last_trading_day(date):
    last_day = date - timedelta(days=1)
    while last_day.weekday() in [5, 6]:  # 5=周六，6=周日
        last_day -= timedelta(days=1)
    return last_day

def update_stock_list(force_update=False):
    if os.path.exists(STOCK_LIST_PATH) and not force_update:
        file_mtime = datetime.fromtimestamp(os.path.getmtime(STOCK_LIST_PATH))
        days_since_update = (datetime.now() - file_mtime).days
        if days_since_update < 7:
            logger.info(f"股票列表文件较新（{days_since_update}天前更新），无需重复获取")
            return STOCK_LIST_PATH

    logger.info("开始在线获取全市场股票列表（可能需要1-2分钟）...")
    lg = None
    try:
        lg = bs.login()
        if lg.error_code != "0":
            raise Exception(f"Baostock登录失败：{lg.error_msg}（错误码：{lg.error_code}）")

        max_attempts = 3
        current_date = datetime.now()

        for attempt in range(max_attempts):
            target_date = current_date - timedelta(days=attempt)
            target_date_str = target_date.strftime("%Y-%m-%d")
            logger.info(f"第{attempt + 1}次尝试：获取{target_date_str}的股票列表")

            rs = bs.query_all_stock(day=target_date_str)  # 指定日期
            if rs.error_code != "0":
                logger.warning(f"{target_date_str}获取失败：{rs.error_msg}")
                continue

            stock_list = []
            while (rs.error_code == "0") & rs.next():
                stock_info = rs.get_row_data()
                stock_code = stock_info[0]
                stock_name = stock_info[2]

                market = stock_code.split(".")[0]
                code_num = stock_code.split(".")[1]
                if (market in VALID_MARKETS) and any(code_num.startswith(p) for p in VALID_CODE_PREFIX):
                    stock_list.append({"code": stock_code, "name": stock_name})

            if stock_list:
                logger.info(f"成功获取{len(stock_list)}只沪深A股数据（{target_date_str}）")
                break
        else:
            raise Exception(f"连续{max_attempts}次获取失败，可能是接口维护或网络问题")

        df = pd.DataFrame(stock_list).drop_duplicates(subset=["code"]).sort_values(by=["code"])
        df.to_csv(STOCK_LIST_PATH, index=False, encoding="utf-8")
        logger.info(f"股票列表已更新并保存至：{STOCK_LIST_PATH}（共{len(df)}条记录）")
        return STOCK_LIST_PATH

    except Exception as e:
        if os.path.exists(STOCK_LIST_PATH):
            logger.warning(f"在线获取失败：{str(e)}，将使用本地旧文件")
            return STOCK_LIST_PATH
        else:
            raise Exception(f"在线获取失败且无本地文件：{str(e)}")

    finally:
        if lg:
            bs.logout()

if __name__ == "__main__":
    try:
        update_stock_list(force_update=True)
        print(f"股票列表已成功更新至：{os.path.abspath(STOCK_LIST_PATH)}")
    except Exception as e:
        print(f"更新失败：{str(e)}")
        exit(1)