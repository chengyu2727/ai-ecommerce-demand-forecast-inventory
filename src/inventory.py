"""
inventory.py —— 库存优化：安全库存 / 再订货点 / 建议补货量
================================================================
把"预测"翻译成"补多少货"，用的是经典的 (R, S) 周期复核 + 安全库存公式：

    日需求速率      mu_d   = 测试期最后 N 天的平均预测值
    单日预测误差     sigma_d = 该 SKU 验证集残差标准差（向整体池化值收缩，防小样本失真）
    提前期内需求    mu_L   = mu_d × L                      (L = lead_time_days)
    提前期误差      sigma_L = sigma_d × sqrt(L)             (假设日误差独立同分布)
    安全库存        SS     = z × sigma_L                    ★ 需求给的公式
    再订货点        ROP    = mu_L + SS
    目标库存水位    S      = mu_d × (L + R) + SS            (R = 复核周期)
    建议补货量      Q      = max(0, S - 当前在库)          并向上取到最小起订量整数倍

z 由服务水平决定：95% → 1.645（用标准正态分位数计算，不是写死的常数表）。
单一公式假设：需求近似正态、误差独立。真实业务中若 SKU 需求高度间歇（长尾），
应改用 bootstrap 或经验分位数法（README 的"待改进"一节有说明）。

运行：
    python -m src.inventory
输出：
    results/replenishment_table.csv
    results/figures/replenishment_top.png
"""

from __future__ import annotations

import json
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import config

OUT_COLS = ["sku", "forecast_demand", "safety_stock", "reorder_point", "suggested_order_qty"]


# ---------------------------------------------------------------------------
# 基础函数
# ---------------------------------------------------------------------------
def z_value(service_level: float = config.SERVICE_LEVEL) -> float:
    """服务水平 → 正态分位数 z（95% → 1.6449）"""
    sl = float(min(max(service_level, 0.5), 0.9999))
    return round(NormalDist().inv_cdf(sl), 4)


def estimate_sigma(fc_valid: pd.DataFrame, shrink: float = config.SIGMA_SHRINK_WEIGHT) -> pd.DataFrame:
    """
    估计每个 SKU 的日预测误差标准差 sigma_d。

    做法：SKU 自身残差标准差 与 全体池化标准差 做方差加权收缩。
    原因：验证期只有 20 多天，单 SKU 标准差极不稳定（长尾 SKU 尤其明显），
          收缩能显著降低安全库存的抖动。
    """
    df = fc_valid.dropna(subset=["residual"]).copy()
    pooled_var = float(df["residual"].var(ddof=1))

    rows = []
    for sku, g in df.groupby("sku_id"):
        n = len(g)
        if n >= 5:
            sku_var = float(g["residual"].var(ddof=1))
        else:
            sku_var = pooled_var
        var = (1 - shrink) * sku_var + shrink * pooled_var
        rows.append({"sku_id": sku, "sigma_daily": float(np.sqrt(max(var, 0.0))),
                     "n_valid_days": n, "sku_sigma": float(np.sqrt(sku_var))})
    out = pd.DataFrame(rows)
    out.attrs["pooled_sigma"] = float(np.sqrt(pooled_var))
    return out


def daily_rate(fc_test: pd.DataFrame, window: int = config.DEMAND_WINDOW_FOR_RATE) -> pd.DataFrame:
    """用测试期最后 window 天的平均预测，作为补货用的日需求速率 mu_d"""
    last_date = fc_test["date"].max()
    start = last_date - pd.Timedelta(days=window - 1)
    recent = fc_test[fc_test["date"] >= start]
    rate = (recent.groupby("sku_id")["y_pred_model"].mean().rename("avg_daily_forecast").reset_index())
    return rate


def latest_state() -> pd.DataFrame:
    """取每个 SKU 最新一天的在库、提前期、品类（现实中来自 WMS / ERP 快照）"""
    clean = pd.read_csv(config.CLEAN_FILE, encoding="utf-8-sig", parse_dates=["date"])
    last_date = clean["date"].max()
    latest = clean[clean["date"] == last_date][
        ["sku_id", "category", "sku_pattern", "on_hand_inventory", "lead_time_days", "price"]
    ].copy()
    return latest


# ---------------------------------------------------------------------------
# 主计算
# ---------------------------------------------------------------------------
def build_table(fc_test: pd.DataFrame, fc_valid: pd.DataFrame,
                service_level: float = config.SERVICE_LEVEL,
                review_days: int = config.REVIEW_PERIOD_DAYS,
                order_multiple: int = config.ORDER_MULTIPLE) -> pd.DataFrame:
    """按 SKU 计算安全库存、再订货点、建议补货量"""
    z = z_value(service_level)
    sigma = estimate_sigma(fc_valid)
    rate = daily_rate(fc_test)
    state = latest_state()

    df = state.merge(sigma[["sku_id", "sigma_daily", "n_valid_days"]], on="sku_id", how="left")
    df = df.merge(rate, on="sku_id", how="left")
    df["avg_daily_forecast"] = df["avg_daily_forecast"].fillna(0.0)
    df["sigma_daily"] = df["sigma_daily"].fillna(df["sigma_daily"].median())

    L = df["lead_time_days"].astype(float)
    mu_d = df["avg_daily_forecast"]

    # 公式主体
    df["forecast_demand"] = (mu_d * L).round(2)                 # 提前期内预测需求 mu_L
    df["sigma_lead_time"] = (df["sigma_daily"] * np.sqrt(L)).round(3)  # sigma_L
    df["safety_stock"] = np.ceil(z * df["sigma_lead_time"]).astype(int)         # SS = z × sigma_L
    df["reorder_point"] = np.ceil(mu_d * L + df["safety_stock"]).astype(int)    # ROP = mu_L + SS
    df["order_up_to_level"] = np.ceil(mu_d * (L + review_days) + df["safety_stock"]).astype(int)

    # 建议补货量：补到目标水位，并向上取整到最小起订倍数
    raw_qty = df["order_up_to_level"] - df["on_hand_inventory"]
    multiples = max(int(order_multiple), 1)
    qty = np.ceil(np.clip(raw_qty, 0, None) / multiples) * multiples
    df["suggested_order_qty"] = qty.astype(int)

    # 附加的业务可解释字段
    df["days_of_cover"] = np.where(mu_d > 0, (df["on_hand_inventory"] / mu_d).round(1), np.nan)
    df["need_order_now"] = (df["on_hand_inventory"] < df["reorder_point"]).astype(int)
    df["z_value"] = z
    df["service_level"] = round(float(service_level), 4)
    df["review_period_days"] = int(review_days)

    df = df.rename(columns={"sku_id": "sku"})
    ordered = OUT_COLS + [
        "category", "sku_pattern", "lead_time_days", "on_hand_inventory",
        "avg_daily_forecast", "sigma_daily", "sigma_lead_time", "order_up_to_level",
        "days_of_cover", "need_order_now", "z_value", "service_level",
        "review_period_days", "n_valid_days", "price",
    ]
    df = df[ordered].sort_values(["need_order_now", "suggested_order_qty"], ascending=[False, False])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------
def plot_top_replenishment(df: pd.DataFrame, top_n: int = 15) -> None:
    """补货量 Top N 的条形图（标注再订货点）"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    top = df.sort_values("suggested_order_qty", ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.barh(top["sku"], top["suggested_order_qty"], color="#3f7fbf", label="suggested order qty")
    ax.scatter(top["reorder_point"], top["sku"], color="#e4572e", s=24, zorder=5, label="reorder point")
    ax.set_xlabel("units", fontsize=9)
    ax.set_title(f"Top {top_n} SKUs by suggested replenishment quantity", fontsize=10, loc="left")
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "replenishment_top.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def run() -> pd.DataFrame:
    if not config.FORECAST_TEST_FILE.exists() or not config.FORECAST_VALID_FILE.exists():
        from . import train
        train.run()

    fc_test = pd.read_csv(config.FORECAST_TEST_FILE, encoding="utf-8-sig", parse_dates=["date"])
    fc_valid = pd.read_csv(config.FORECAST_VALID_FILE, encoding="utf-8-sig", parse_dates=["date"])

    table = build_table(fc_test, fc_valid)
    table.to_csv(config.REPLENISHMENT_FILE, index=False, encoding="utf-8-sig")

    # 参数快照（可复现：一眼看出表是用什么口径算的）
    snapshot = {
        "service_level": config.SERVICE_LEVEL,
        "z_value": z_value(config.SERVICE_LEVEL),
        "review_period_days": config.REVIEW_PERIOD_DAYS,
        "order_multiple": config.ORDER_MULTIPLE,
        "demand_window_for_rate": config.DEMAND_WINDOW_FOR_RATE,
        "sigma_method": "验证集残差标准差，向池化标准差方差收缩（权重 %.2f）" % config.SIGMA_SHRINK_WEIGHT,
        "formulas": {
            "safety_stock": "z * sigma_daily * sqrt(lead_time_days)",
            "reorder_point": "avg_daily_forecast * lead_time_days + safety_stock",
            "suggested_order_qty": "max(0, avg_daily_forecast*(lead_time+review) + SS - on_hand)，向上取整到起订倍数",
        },
        "n_skus": int(len(table)),
        "total_suggested_qty": int(table["suggested_order_qty"].sum()),
        "n_need_order_now": int(table["need_order_now"].sum()),
        "data_as_of": str(pd.read_csv(config.CLEAN_FILE, encoding="utf-8-sig", usecols=["date"])["date"].max()),
    }
    (config.RESULTS / "inventory_params.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[inventory] 服务水平 {config.SERVICE_LEVEL:.0%} → z = {snapshot['z_value']} | "
          f"复核周期 {config.REVIEW_PERIOD_DAYS} 天")
    print(f"  需要立即补货的 SKU = {snapshot['n_need_order_now']} / {snapshot['n_skus']} | "
          f"建议补货总量 = {snapshot['total_suggested_qty']:,} 件")
    show = ["sku", "forecast_demand", "safety_stock", "reorder_point",
            "on_hand_inventory", "suggested_order_qty", "days_of_cover", "lead_time_days"]
    print("\n  补货量 Top 10：")
    print(table.head(10)[show].to_string(index=False))
    print(f"\n  → {config.REPLENISHMENT_FILE}")

    plot_top_replenishment(table)
    return table


if __name__ == "__main__":
    run()
