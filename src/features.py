"""
features.py —— 特征工程 + 严格按时间切分
================================================================
特征分四组（对应需求）：
  1) 滞后特征  lag_1 / lag_7 / lag_14     —— 捕捉惯性、周内重复性
  2) 滑动窗口  rolling_7/14/30 mean & std  —— 捕捉近期水位与波动
  3) 日历特征  weekday / month / day_of_month / is_weekend
  4) 业务标记  promo_flag / holiday_flag / promo_days_last_7 / price / price_ratio

两条铁律（作品集里必须写清楚）：
  * 所有滞后/滑窗都先 shift(1)，绝不允许用到"当天"的销量（否则就是未来信息泄漏）；
  * 训练/验证/测试严格按日期切分，不是随机切分 —— 时序预测随机切分等于作弊。

关于断货截断：lag/rolling 只能基于"观测到的销量"，断货日的 0 会被当作真实低销量，
这本身是行业里的真实困难。做法是：训练时剔除 is_censored==1 的行（目标被截断，标签不可信），
在评估时单独统计断货占比。真实项目里还可以进一步做截断需求还原（censored demand recovery）。

运行：
    python -m src.features
输出：
    data/processed/features.csv
    data/processed/split_info.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config

# 模型真正使用的特征列（不含目标、不含任何 id 原文、不含模拟标签）
FEATURE_COLS = [
    # 滞后
    "lag_1", "lag_7", "lag_14",
    # 滑动窗口
    "rolling_7_mean", "rolling_7_std", "rolling_14_mean",
    "rolling_30_mean", "rolling_30_std", "r7_over_r30",
    # 日历
    "weekday", "month", "day_of_month", "is_weekend",
    # 业务标记
    "promo_flag", "promo_days_last_7", "holiday_flag",
    "price", "price_ratio", "lead_time_days",
    # 类别编码
    "sku_code", "category_code",
]

CATEGORICAL_COLS = ["sku_code", "category_code"]
TARGET_COL = config.TARGET


# ---------------------------------------------------------------------------
# 各组特征
# ---------------------------------------------------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """日历特征：星期、月份、月内第几天、是否周末"""
    df = df.copy()
    df["weekday"] = df["date"].dt.day_of_week.astype(int)      # 0=周一
    df["month"] = df["date"].dt.month.astype(int)
    df["day_of_month"] = df["date"].dt.day.astype(int)
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """滞后 + 滑动窗口特征（全部按 SKU 分组、shift(1) 防泄漏）"""
    df = df.copy()
    df = df.sort_values(["sku_id", "date"])

    grp = df.groupby("sku_id", sort=False)[TARGET_COL]

    # 滞后：昨天 / 上周同一天 / 两周前同一天
    for lag in (1, 7, 14):
        df[f"lag_{lag}"] = grp.shift(lag)

    # 滑动窗口：都用 shift(1) 后滚动，等于"截至昨天的最近 N 天"
    mp = config.ROLLING_MIN_PERIODS
    lagged = grp.shift(1)
    for win in (7, 14, 30):
        roll = lagged.groupby(df["sku_id"]).rolling(win, min_periods=min(mp, win))
        df[f"rolling_{win}_mean"] = roll.mean().reset_index(level=0, drop=True)
        df[f"rolling_{win}_std"] = roll.std().reset_index(level=0, drop=True)

    # 短期/中期水位比：>1 说明近期在放量
    df["r7_over_r30"] = df["rolling_7_mean"] / df["rolling_30_mean"].replace(0, np.nan)

    # 过去 7 天促销天数（促销有持续影响，比单日 flag 更有信息量）
    df["promo_days_last_7"] = (
        df.groupby("sku_id", sort=False)["promo_flag"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=1).sum())
    )
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """价格特征：绝对价格 + 相对该 SKU 常态价的折扣力度"""
    df = df.copy()
    median_price = df.groupby("sku_id")["price"].transform("median")
    df["price_ratio"] = (df["price"] / median_price.replace(0, np.nan)).astype(float)
    return df


def add_categorical_codes(df: pd.DataFrame) -> pd.DataFrame:
    """把 sku_id / category 编码成整数（树模型用整数编码 + 声明为类别特征）"""
    df = df.copy()
    sku_uniques = np.sort(df["sku_id"].unique())        # 排序后编码，保证任何机器上编码一致
    cat_uniques = np.sort(df["category"].unique())
    mapping_sku = {v: i for i, v in enumerate(sku_uniques)}
    mapping_cat = {v: i for i, v in enumerate(cat_uniques)}
    df["sku_code"] = df["sku_id"].map(mapping_sku).astype(int)
    df["category_code"] = df["category"].map(mapping_cat).astype(int)
    return df


def build(df: pd.DataFrame) -> pd.DataFrame:
    """特征工程总入口"""
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_price_features(df)
    df = add_categorical_codes(df)

    # 断货截断标记：训练时要剔除这些行（标签不可信）
    df["is_censored"] = df["stockout_flag"].astype(int)
    return df


# ---------------------------------------------------------------------------
# 时间切分
# ---------------------------------------------------------------------------
def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    严格按日期切分训练/验证/测试（同一日期对所有 SKU 属于同一个集合，避免跨期泄漏）。
    前 WARMUP_DAYS 天因为 lag/rolling 缺失，标记为 warmup，不参与训练与评估。
    """
    df = df.copy()
    dates = np.sort(df["date"].unique())
    n = len(dates)
    n_train = int(n * config.TRAIN_RATIO)
    n_valid = int(n * config.VALID_RATIO)

    train_dates = dates[:n_train]
    valid_dates = dates[n_train:n_train + n_valid]
    test_dates = dates[n_train + n_valid:]

    df["split"] = "test"
    df.loc[df["date"].isin(valid_dates), "split"] = "valid"
    df.loc[df["date"].isin(train_dates), "split"] = "train"

    # 特征不完整的窗口日单独标出（这些行不参与训练/评估）
    incomplete = df["rolling_30_mean"].isna() | df["lag_14"].isna()
    n_incomplete_in_train = int((incomplete & (df["split"] == "train")).sum())
    df.loc[incomplete & (df["split"] == "train"), "split"] = "warmup"

    info = {
        "train": [str(pd.Timestamp(train_dates[0]).date()), str(pd.Timestamp(train_dates[-1]).date())],
        "valid": [str(pd.Timestamp(valid_dates[0]).date()), str(pd.Timestamp(valid_dates[-1]).date())],
        "test": [str(pd.Timestamp(test_dates[0]).date()), str(pd.Timestamp(test_dates[-1]).date())],
        "n_days": {"train": int(len(train_dates)), "valid": int(len(valid_dates)), "test": int(len(test_dates))},
        "rows": df["split"].value_counts().to_dict(),
        "warmup_rows_moved_out_of_train": n_incomplete_in_train,
        "note": "按日期切分，同日期的全部 SKU 属于同一集合；warmup 行不参与训练与评估",
    }
    info["rows"] = {k: int(v) for k, v in info["rows"].items()}
    return df, info


def load_features(path=config.FEATURE_FILE) -> pd.DataFrame:
    """读取已生成的特征表（供 train / evaluate / inventory / notebook 复用）"""
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}，请先运行：python -m src.features")
    df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"])
    return df


def run(force: bool = False) -> pd.DataFrame:
    """入口：清洗表 → 特征 → 时间切分 → 落盘"""
    if not config.CLEAN_FILE.exists() or force:
        from . import data_prep
        data_prep.run()

    clean_df = pd.read_csv(config.CLEAN_FILE, encoding="utf-8-sig", parse_dates=["date"])
    feats = build(clean_df)
    feats, info = time_split(feats)
    feats.to_csv(config.FEATURE_FILE, index=False, encoding="utf-8-sig")
    with open(config.SPLIT_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"[features] 特征表已写入 {config.FEATURE_FILE}")
    print(f"  特征数 = {len(FEATURE_COLS)} | 行数 = {len(feats):,}")
    print(f"  训练集 {info['train'][0]} ~ {info['train'][1]}（{info['n_days']['train']} 天，"
          f"{info['rows'].get('train', 0):,} 行）")
    print(f"  验证集 {info['valid'][0]} ~ {info['valid'][1]}（{info['n_days']['valid']} 天，"
          f"{info['rows'].get('valid', 0):,} 行）")
    print(f"  测试集 {info['test'][0]} ~ {info['test'][1]}（{info['n_days']['test']} 天，"
          f"{info['rows'].get('test', 0):,} 行）")
    print(f"  warmup 行（特征不完整，不参与训练）= {info['rows'].get('warmup', 0):,}")
    return feats


if __name__ == "__main__":
    run()
