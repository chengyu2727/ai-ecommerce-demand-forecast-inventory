"""
data_generator.py —— 生成模拟电商日粒度销量数据
================================================================
目的：在还没有真实业务数据时，先造一份"结构上像真实电商"的数据，
      让后续的特征工程、预测、库存优化都能端到端跑通。

关键设计（都是为了不把问题做简单）：
1. SKU × 日期 粒度，最近 N_DAYS 天，N_SKUS 个 SKU；
2. 四种需求模式：stable 稳定型 / promo 促销型 / seasonal 季节型 / longtail 长尾型；
3. 促销用"马尔可夫链"生成成簇的促销期，并有促销后的需求回落（提前购买效应）；
4. 节假日按真实法定节假日区间打标，并按模式给不同的抬升系数；
5. 需求生成为泊松 / 负二项分布（后者更过度离散），不是简单加高斯噪声；
6. 模拟一个"朴素补货策略"的零售库存系统，从而自然地产生断货：
   断货日 units_sold 会被截断（可能为 0），同时 stockout_flag=1，
   并用 demand_latent 保留"真实需求"以验证截断处理是否正确
   —— 注意 demand_latent 只有模拟数据才有，真实业务中不可观测，绝不能当特征。

运行：
    python -m src.data_generator
输出：
    data/raw/sales_daily.csv
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import config

# ---------------------------------------------------------------------------
# 各模式的"性格"参数
# ---------------------------------------------------------------------------
# 周一 → 周日的星期系数
WEEKDAY_PATTERN = {
    "stable":   [1.00, 1.00, 1.00, 1.00, 1.03, 1.08, 1.04],
    "promo":    [0.95, 0.95, 0.98, 1.00, 1.05, 1.15, 1.10],
    "seasonal": [0.85, 0.88, 0.92, 0.98, 1.10, 1.35, 1.28],
    "longtail": [1.00, 1.00, 1.00, 1.00, 1.00, 1.02, 1.00],
}

# 促销日相对日常的销量倍数区间
PROMO_MULT_RANGE = {
    "stable":   (2.0, 3.0),
    "promo":    (3.0, 5.5),
    "seasonal": (1.8, 2.8),
    "longtail": (2.0, 4.0),
}

# 节假日抬升系数
HOLIDAY_MULT = {"stable": 1.08, "promo": 1.25, "seasonal": 1.35, "longtail": 1.10}

# 促销概率（马尔可夫：昨天没促 → 今天开始促的概率；昨天在促 → 今天继续的概率）
PROMO_START_P = {"stable": 0.015, "promo": 0.085, "seasonal": 0.045, "longtail": 0.010}
PROMO_CONT_P = 0.72
PROMO_MAX_RUN = 4

# 促销后的需求回落（提前购买），持续 2 天
POST_PROMO_DIP = 0.75
POST_PROMO_DAYS = 2


@dataclass
class SkuProfile:
    """单个 SKU 的生成参数（都把随机性收敛到这里，方便复现）"""

    sku_id: str
    category: str
    pattern: str
    base_level: float                     # 日常日均需求
    promo_mult: float                     # 促销倍数
    price0: float                         # 常规价
    lead_time_days: int                   # 供应商提前期
    trend: float                          # 每日对数趋势（±）
    weekday_mult: np.ndarray = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def build_date_range(n_days: int = config.N_DAYS, end: date = config.ANCHOR_DATE) -> list[date]:
    """返回长度为 n_days 的日期列表（升序，含 end）"""
    return [end - timedelta(days=i) for i in range(n_days - 1, -1, -1)]


def build_holiday_flags(dates: list[date]) -> tuple[np.ndarray, np.ndarray]:
    """生成 holiday_flag（是否法定节假日）与 near_holiday_flag（节前/节后 1 天）"""
    holiday_dates: set[date] = set()
    for start, end in config.HOLIDAY_RANGES:
        d = start
        while d <= end:
            holiday_dates.add(d)
            d += timedelta(days=1)

    flag = np.array([int(d in holiday_dates) for d in dates], dtype=int)
    near = np.zeros(len(dates), dtype=int)
    w = config.HOLIDAY_LIFT_WINDOW
    if w > 0:
        for i, d in enumerate(dates):
            if d in holiday_dates:
                continue
            for off in range(1, w + 1):
                if (d + timedelta(days=off)) in holiday_dates or (d - timedelta(days=off)) in holiday_dates:
                    near[i] = 1
                    break
    return flag, near


def _seasonal_wave(t: np.ndarray, pattern: str) -> np.ndarray:
    """季节性波动：季节型叠加 90 天中周期 + 30 天短周期；其余模式只有很弱的漂移"""
    if pattern == "seasonal":
        return 1.0 + 0.35 * np.sin(2 * np.pi * t / 90.0 + 0.6) + 0.12 * np.sin(2 * np.pi * t / 30.0)
    if pattern == "stable":
        return 1.0 + 0.05 * np.sin(2 * np.pi * t / 60.0)
    if pattern == "promo":
        return 1.0 + 0.08 * np.sin(2 * np.pi * t / 45.0)
    return np.ones_like(t, dtype=float)


def _promo_series(n: int, pattern: str, rng: np.random.Generator) -> np.ndarray:
    """马尔可夫链生成成簇的促销日历（0/1）"""
    p_start = PROMO_START_P[pattern]
    promo = np.zeros(n, dtype=int)
    run = 0
    for t in range(n):
        if run > 0:
            if rng.random() < PROMO_CONT_P and run < PROMO_MAX_RUN:
                promo[t] = 1
                run += 1
            else:
                run = 0
        else:
            if rng.random() < p_start:
                promo[t] = 1
                run = 1
    return promo


def _make_profiles(rng: np.random.Generator) -> list[SkuProfile]:
    """构造 N_SKUS 个 SKU 画像：模式按顺序轮转，保证四种模式都有足够样本"""
    categories = list(config.CATEGORIES)
    lead_time_choices = np.array([3, 5, 7, 10, 14])
    lead_time_p = np.array([0.25, 0.30, 0.25, 0.13, 0.07])

    profiles: list[SkuProfile] = []
    for i in range(config.N_SKUS):
        pattern = config.PATTERNS[i % len(config.PATTERNS)]
        category = categories[i % len(categories)]
        lo, hi = config.CATEGORIES[category]

        # 各模式的日常量级差异很大：长尾 SKU 一天可能卖不到 1 件
        level_range = {
            "stable": (45.0, 120.0),
            "promo": (18.0, 55.0),
            "seasonal": (30.0, 95.0),
            "longtail": (0.6, 3.5),
        }[pattern]
        base_level = float(rng.uniform(*level_range))

        lo_m, hi_m = PROMO_MULT_RANGE[pattern]
        profiles.append(
            SkuProfile(
                sku_id=f"SKU{i + 1:03d}",
                category=category,
                pattern=pattern,
                base_level=base_level,
                promo_mult=float(rng.uniform(lo_m, hi_m)),
                price0=float(np.round(rng.uniform(lo, hi), 1)) - 0.1,  # 尾数 .9 更像零售价
                lead_time_days=int(rng.choice(lead_time_choices, p=lead_time_p)),
                trend=float(rng.normal(0.0, 0.0008)),  # 180 天累计约 ±15%
                weekday_mult=np.array(WEEKDAY_PATTERN[pattern], dtype=float),
            )
        )
    return profiles


def _draw_demand(mu: np.ndarray, pattern: str, rng: np.random.Generator) -> np.ndarray:
    """按均值 mu 抽样整数需求。促销/长尾用负二项分布制造过度离散（大促尖峰、偶发大单）"""
    mu = np.clip(mu, 0.0, None)
    if pattern in ("promo", "longtail"):
        # 负二项：均值 mu，方差 mu + mu^2/r（r 越小越离散）
        r = 6.0
        p = r / (r + mu)
        return rng.negative_binomial(r, p).astype(int)
    return rng.poisson(mu).astype(int)


def _simulate_inventory(
    demand: np.ndarray, lead_time: int, rng: np.random.Generator, review_days: int = 7
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    模拟一个"朴素补货策略"的库存系统，制造出真实的断货截断。

    策略：库存头寸（在库 + 在途）跌破补货点就下单，lead_time 天后到货。
    返回 (units_sold, on_hand_inventory, stockout_flag)：
        units_sold         实际卖出量 = min(需求, 可售库存)  → 断货时被截断，可能为 0
        stockout_flag      需求未被完全满足 = 1（含部分断货；完全无货时 units_sold=0）
    """
    n = len(demand)
    mean_d = max(float(demand.mean()), 0.5)
    sd_d = max(float(demand.std()), 0.5)

    # 零售商的主观补货点（故意设得偏乐观，才会断货）
    reorder_point = mean_d * lead_time + 0.8 * sd_d * np.sqrt(lead_time)
    order_qty = max(int(round(mean_d * review_days * 1.2)), 1)

    arrivals = np.zeros(n + lead_time + 1, dtype=float)
    on_hand = mean_d * (lead_time + review_days) * float(rng.uniform(0.8, 1.3))
    on_order = 0.0

    sold = np.zeros(n, dtype=int)
    on_hand_hist = np.zeros(n, dtype=int)
    stockout = np.zeros(n, dtype=int)

    for t in range(n):
        if arrivals[t] > 0:                      # 当天到货
            on_hand += arrivals[t]
            on_order -= arrivals[t]

        d = int(demand[t])
        avail = on_hand
        sell = min(d, avail)                     # 卖不出去的部分就是被截断的需求
        sold[t] = int(sell)
        stockout[t] = int(d > avail + 1e-9)
        on_hand = avail - sell

        # 收盘盘点：库存头寸不足则补货
        if on_hand + on_order <= reorder_point:
            arrivals[t + lead_time] += order_qty
            on_order += order_qty

        on_hand_hist[t] = int(np.floor(on_hand))

    return sold, on_hand_hist, stockout


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def generate(
    n_days: int = config.N_DAYS,
    n_skus: int = config.N_SKUS,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """生成完整的模拟数据集，返回长表 DataFrame"""
    rng = np.random.default_rng(seed)
    dates = build_date_range(n_days)
    holiday_flag, near_holiday = build_holiday_flags(dates)
    t_idx = np.arange(n_days, dtype=float)

    profiles = _make_profiles(rng)
    # 每个 SKU 用独立子种子：加 SKU 时不会改变已有 SKU 的数据
    child_seeds = rng.integers(0, 2**31 - 1, size=n_skus)

    frames: list[pd.DataFrame] = []
    for prof, sub_seed in zip(profiles, child_seeds):
        sub_rng = np.random.default_rng(int(sub_seed))

        # 1) 促销日历
        promo = _promo_series(n_days, prof.pattern, sub_rng)
        # 促销后回落：促销结束后 POST_PROMO_DAYS 天
        post_dip = np.zeros(n_days, dtype=int)
        for t in range(n_days - 1):
            if promo[t] == 1 and promo[t + 1] == 0:
                for k in range(1, POST_PROMO_DAYS + 1):
                    if t + k < n_days:
                        post_dip[t + k] = 1

        # 2) 需求均值 = 基准 × 星期 × 季节 × 促销 × 节假日 × 趋势
        mu = np.full(n_days, prof.base_level, dtype=float)
        dow = np.array([d.weekday() for d in dates])
        mu *= prof.weekday_mult[dow]
        mu *= _seasonal_wave(t_idx, prof.pattern)
        mu *= np.where(promo == 1, prof.promo_mult, 1.0)
        mu *= np.where(post_dip == 1, POST_PROMO_DIP, 1.0)
        mu *= np.where(holiday_flag == 1, HOLIDAY_MULT[prof.pattern], 1.0)
        mu *= np.where((near_holiday == 1) & (holiday_flag == 0), 1.0 + 0.6 * (HOLIDAY_MULT[prof.pattern] - 1.0), 1.0)
        mu *= np.exp(prof.trend * t_idx)
        # 长尾 SKU 加偶发爆量日（真实长尾商品的典型特征）
        if prof.pattern == "longtail":
            spikes = sub_rng.random(n_days) < 0.04
            mu = np.where(spikes, mu * 8.0, mu)

        # 3) 抽真实需求
        demand_latent = _draw_demand(mu, prof.pattern, sub_rng)

        # 4) 库存仿真 → 观测销量（断货截断）
        units_sold, on_hand, stockout = _simulate_inventory(demand_latent, prof.lead_time_days, sub_rng)

        # 5) 价格：促销日打折，其余日子小幅波动
        price_noise = sub_rng.normal(1.0, 0.015, n_days)
        price = prof.price0 * price_noise * np.where(promo == 1, 0.80, 1.0)
        price = np.round(np.clip(price, 1.0, None), 2)

        frames.append(
            pd.DataFrame(
                {
                    "sku_id": prof.sku_id,
                    "date": dates,
                    "category": prof.category,
                    "price": price,
                    "promo_flag": promo,
                    "holiday_flag": holiday_flag,
                    "units_sold": units_sold,
                    "on_hand_inventory": on_hand,
                    "lead_time_days": prof.lead_time_days,
                    # 以下两列是模拟数据独有的，真实数据拿不到（见 README）
                    "stockout_flag": stockout,
                    "demand_latent": demand_latent,
                    "sku_pattern": prof.pattern,
                }
            )
        )

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["sku_id", "date"]).reset_index(drop=True)


def main() -> None:
    df = generate()
    df.to_csv(config.RAW_FILE, index=False, encoding="utf-8-sig")

    n_days = df["date"].nunique()
    print(f"[data_generator] 已生成 {config.RAW_FILE}")
    print(f"  行数 = {len(df):,} | SKU = {df['sku_id'].nunique()} | 日期跨度 = {n_days} 天 "
          f"({df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d})")
    print(f"  断货行占比 = {df['stockout_flag'].mean():.2%} | 断货且销量为 0 的行数 = "
          f"{int(((df['stockout_flag'] == 1) & (df['units_sold'] == 0)).sum())}")
    summary = (
        df.groupby("sku_pattern")
        .agg(日均销量=("units_sold", "mean"), 日需求真实均值=("demand_latent", "mean"),
             断货率=("stockout_flag", "mean"), SKU数=("sku_id", "nunique"))
        .round(3)
    )
    print("\n按需求模式的汇总：")
    print(summary.to_string())


if __name__ == "__main__":
    main()
