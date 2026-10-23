"""
data_prep.py —— 数据清洗与质量校验
================================================================
职责：把"原始表"变成"可直接做特征工程的干净表"，只做与时间序列结构相关的处理：
1. 类型统一、去重（同一 SKU 同一天只保留一条）；
2. 按 SKU 补齐缺失日期：时间序列一旦缺行，lag/rolling 就会错位，
   所以必须显式补成连续日历（补出来的行销量记 0，并单独统计条数）；
3. 价格 / 品类 / 提前期缺失用同 SKU 前后值填充（ffill → bfill）；
4. 异常值：负销量、负库存截断为 0；销量强制取整；
5. 输出质量报告，把"数据有多少坑"打印出来（写进作品集时这就是证据）。

不做的事：不删离群点、不改变量纲、不对目标做平滑 —— 这些都会掩盖真实问题。

运行：
    python -m src.data_prep
输出：
    data/processed/sales_clean.csv
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config

REQUIRED_COLS = [
    "sku_id", "date", "category", "price", "promo_flag", "holiday_flag",
    "units_sold", "on_hand_inventory", "lead_time_days", "stockout_flag",
]

FLAG_COLS = ["promo_flag", "holiday_flag", "stockout_flag"]


def load_raw(path=config.RAW_FILE) -> pd.DataFrame:
    """读取原始数据（保留 utf-8-sig，兼容 Excel 打开）"""
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}，请先运行：python -m src.data_generator")
    return pd.read_csv(path, encoding="utf-8-sig")


def _check_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"原始数据缺少必需列：{missing}")


def _fill_missing_dates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """按 sku_id 补齐缺失日期，返回 (新表, 补充的行数)"""
    full_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    pieces = []
    for sku, g in df.groupby("sku_id", sort=True):
        g = g.set_index("date").reindex(full_dates)
        n_missing = int(g["sku_id"].isna().sum())

        # 静态/可继承字段：前后填充
        g["sku_id"] = sku
        for col in ("category", "lead_time_days"):
            g[col] = g[col].ffill().bfill()
        g["price"] = g["price"].ffill().bfill()
        g["on_hand_inventory"] = g["on_hand_inventory"].ffill().fillna(0)

        # 业务字段：补出来的日子没有销售、没有促销、没有断货记录
        g["units_sold"] = g["units_sold"].fillna(0)
        for col in FLAG_COLS:
            g[col] = g[col].fillna(0)
        for col in ("demand_latent", "sku_pattern"):
            if col in g.columns:
                g[col] = g[col].fillna(g[col].ffill())

        g.index.name = "date"
        pieces.append(g.reset_index())
    out = pd.concat(pieces, ignore_index=True)
    n_added = int(len(out) - len(df))
    return out, n_added


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """主清洗流程（幂等，可重复执行）"""
    _check_schema(df)
    df = df.copy()

    # 1) 类型统一
    df["date"] = pd.to_datetime(df["date"])
    df["sku_id"] = df["sku_id"].astype(str)
    df["category"] = df["category"].astype(str)
    for col in FLAG_COLS + ["lead_time_days"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in ("price", "units_sold", "on_hand_inventory"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2) 去重：同 SKU 同日期保留最后一条（通常是最新修正）
    df = df.sort_values(["sku_id", "date"]).drop_duplicates(["sku_id", "date"], keep="last")

    # 3) 补齐连续日历（时间序列的关键一步）
    df, n_filled = _fill_missing_dates(df)
    df.attrs["n_filled_rows"] = n_filled

    # 4) 异常值与口径修正
    df["units_sold"] = df["units_sold"].clip(lower=0).fillna(0).round().astype(int)
    df["on_hand_inventory"] = df["on_hand_inventory"].clip(lower=0).fillna(0).round().astype(int)
    df["price"] = df["price"].clip(lower=0.01)
    df["lead_time_days"] = df["lead_time_days"].clip(lower=1, upper=90)

    # 5) 目标列缺失（真没卖 = 0），但历史里若有"未记录"需单独处理，这里按 0 并记录
    df["units_sold"] = df["units_sold"].fillna(0).astype(int)

    # 6) 衍生两个常用的"是/否"字段，避免后面重复写表达式
    df["is_weekend"] = (df["date"].dt.day_of_week >= 5).astype(int)
    df["weekday"] = df["date"].dt.day_of_week.astype(int)

    return df.sort_values(["sku_id", "date"]).reset_index(drop=True)


def quality_report(df: pd.DataFrame) -> dict:
    """生成数据质量报告（打印 + 返回 dict，可写入 json）"""
    report = {
        "rows": int(len(df)),
        "n_skus": int(df["sku_id"].nunique()),
        "n_categories": int(df["category"].nunique()),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "n_days": int(df["date"].nunique()),
        "filled_rows": int(df.attrs.get("n_filled_rows", 0)),
        "stockout_rate": round(float(df["stockout_flag"].mean()), 4),
        "zero_sales_rate": round(float((df["units_sold"] == 0).mean()), 4),
        "promo_rate": round(float(df["promo_flag"].mean()), 4),
        "holiday_rate": round(float(df["holiday_flag"].mean()), 4),
        "duplicated_sku_date": int(df.duplicated(["sku_id", "date"]).sum()),
        "negative_sales": int((df["units_sold"] < 0).sum()),
        "price_missing": int(df["price"].isna().sum()),
        # 时间连续性检查：每个 SKU 的行数应等于日期跨度的天数
        "continuous_dates": bool(
            (df.groupby("sku_id")["date"].count() == df["date"].nunique()).all()
        ),
    }

    print(f"[data_prep] 清洗完成：{report['rows']:,} 行 / {report['n_skus']} SKU / "
          f"{report['n_days']} 天 ({report['date_min']} ~ {report['date_max']})")
    print(f"  补齐缺失日期行数 = {report['filled_rows']} | 日期连续 = {report['continuous_dates']}")
    print(f"  断货率 = {report['stockout_rate']:.2%} | 零销量占比 = {report['zero_sales_rate']:.2%} | "
          f"促销日占比 = {report['promo_rate']:.2%} | 节假日占比 = {report['holiday_rate']:.2%}")
    if report["zero_sales_rate"] > report["stockout_rate"] + 0.05:
        print("  提示：零销量占比明显高于断货率，说明存在真实的“没卖动”日子（如长尾 SKU），不是数据缺失。")
    return report


def run() -> pd.DataFrame:
    """入口：读原始表 → 清洗 → 落盘 → 打印质量报告"""
    if not config.RAW_FILE.exists():
        print("[data_prep] 未找到原始数据，自动调用 data_generator 生成…")
        from . import data_generator
        data_generator.main()
    raw = load_raw()
    df = clean(raw)
    report = quality_report(df)
    df.to_csv(config.CLEAN_FILE, index=False, encoding="utf-8-sig")
    with open(config.DATA_PROCESSED / "data_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  → 已写入 {config.CLEAN_FILE}")
    return df


if __name__ == "__main__":
    run()
