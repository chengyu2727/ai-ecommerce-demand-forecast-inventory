"""
evaluate.py —— 预测效果评估
================================================================
四个指标（都是"越小越好"）：
    MAE   平均绝对误差，单位=件，业务方最好理解
    RMSE  均方根误差，对大误差更敏感（惩罚促销尖峰漏预测）
    MAPE  平均绝对百分比误差，跨 SKU 可比，但 y=0 时无定义 → 只在 y>0 的样本上算，并报告占比
    WMAPE 加权 MAPE = Σ|误差| / Σ实际，等价于"整体偏差率"，不受单日小基数放大影响
          —— 快消/电商最常用的汇总指标，作为主指标

评估口径（必须说清楚，否则数字没意义）：
    主指标只在实际未被断货截断的测试行上计算（stockout_flag==0），
    因为截断行的"实际销量"本身就被库存低估了，拿它算误差是在惩罚正确预测。
    断货行占比会单独输出。

运行：
    python -m src.evaluate
输出：
    results/model_metrics.json    模型对比（整体 / 分模式 / 分 SKU 摘要）
    results/metrics_by_sku.csv    每个 SKU 的误差明细
    results/figures/*.png         预测曲线、特征重要性、误差对比
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config

MODEL_COLUMNS = {
    "baseline_last_week": "y_pred_baseline_last_week",
    "baseline_ma_7": "y_pred_baseline_ma_7",
    "model": "y_pred_model",
}

MLABEL = {
    "baseline_last_week": "基线-上周同日",
    "baseline_ma_7": "基线-近7天均值",
    "model": "主模型(GBDT)",
}

# 图里统一用英文标签，避免不同机器缺中文字体时出现方框
PLOT_LABEL = {
    "baseline_last_week": "baseline: last week",
    "baseline_ma_7": "baseline: MA7",
    "model": "GBDT model",
}


# ---------------------------------------------------------------------------
# 指标函数（纯函数，方便单测）
# ---------------------------------------------------------------------------
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE（%）。约定：真实值为 0 的样本无法计算百分比误差，直接剔除。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """WMAPE（%）= Σ|误差| / Σ实际值，分母为 0 时返回 nan"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denom * 100)


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": round(mae(y_true, y_pred), 4),
        "RMSE": round(rmse(y_true, y_pred), 4),
        "MAPE_%": round(mape(y_true, y_pred), 4),
        "WMAPE_%": round(wmape(y_true, y_pred), 4),
        "n_samples": int(len(y_true)),
        "n_zero_actual": int((np.asarray(y_true) == 0).sum()),
    }


# ---------------------------------------------------------------------------
# 评估流程
# ---------------------------------------------------------------------------
def evaluate_frame(df: pd.DataFrame, only_uncensored: bool = True) -> dict:
    """对一张预测明细表做整体 / 分模式 评估"""
    data = df[df["stockout_flag"] == 0] if only_uncensored else df
    result: dict = {
        "evaluated_rows": int(len(data)),
        "total_rows": int(len(df)),
        "censored_rows_excluded": int(len(df) - len(data)),
        "censored_rate": round(float(df["stockout_flag"].mean()), 4),
        "metrics": {},
        "by_pattern": {},
    }
    for name, col in MODEL_COLUMNS.items():
        if col in data.columns:
            result["metrics"][name] = all_metrics(data[config.TARGET].to_numpy(), data[col].to_numpy())

    if "sku_pattern" in data.columns:
        for pattern, g in data.groupby("sku_pattern"):
            result["by_pattern"][pattern] = {
                name: all_metrics(g[config.TARGET].to_numpy(), g[col].to_numpy())
                for name, col in MODEL_COLUMNS.items() if col in g.columns
            }
    return result


def per_sku_table(df: pd.DataFrame) -> pd.DataFrame:
    """每个 SKU 的误差明细（看模型在哪类商品上失效）"""
    data = df[df["stockout_flag"] == 0]
    rows = []
    for sku, g in data.groupby("sku_id"):
        row = {
            "sku_id": sku,
            "category": g["category"].iloc[0],
            "sku_pattern": g["sku_pattern"].iloc[0],
            "mean_actual": round(float(g[config.TARGET].mean()), 3),
        }
        for name, col in MODEL_COLUMNS.items():
            if col in g.columns:
                row[f"MAE_{name}"] = round(mae(g[config.TARGET], g[col]), 3)
                row[f"WMAPE_{name}"] = round(wmape(g[config.TARGET], g[col]), 4)
        rows.append(row)
    out = pd.DataFrame(rows)
    if "WMAPE_model" in out.columns:
        out = out.sort_values("WMAPE_model", ascending=False)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------
def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.formatter.use_mathtext"] = False
    # 图中统一用英文标签，避免不同机器缺中文字体导致方框
    return plt


def _format_date_axis(ax, interval: int = 4) -> None:
    """日期轴：短格式 + 固定间隔，避免标签重叠"""
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))


def plot_forecast_curves(df: pd.DataFrame, plt) -> None:
    """每种需求模式挑一个 SKU，画测试期 实际 vs 预测"""
    patterns = [p for p in config.PATTERNS if p in set(df["sku_pattern"])]
    n = len(patterns)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.4 * n), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, pattern in zip(axes, patterns):
        sub = df[df["sku_pattern"] == pattern]
        sku = sorted(sub["sku_id"].unique())[0]
        g = sub[sub["sku_id"] == sku].sort_values("date")
        ax.plot(g["date"], g[config.TARGET], label="actual (observed)", color="#2c7fb8", lw=1.6)
        ax.plot(g["date"], g["y_pred_model"], label="pred (GBDT)", color="#e4572e", lw=1.4, alpha=0.9)
        if "y_pred_baseline_ma_7" in g.columns:
            ax.plot(g["date"], g["y_pred_baseline_ma_7"], label="baseline MA7",
                    color="#9aa5b1", lw=1.0, ls="--")
        so = g[g["stockout_flag"] == 1]
        if len(so):
            ax.scatter(so["date"], so[config.TARGET], s=18, color="#7b2cbf",
                       zorder=5, label="stockout day")
        ax.set_title(f"{sku} · {pattern}", fontsize=10, loc="left")
        ax.set_ylabel("units/day", fontsize=8)
        ax.tick_params(labelsize=7)
        _format_date_axis(ax)
        ax.legend(fontsize=7, frameon=False, ncol=4, loc="upper left")
    fig.suptitle("Test period: actual vs forecast (one SKU per demand pattern)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(config.FIGURES / "forecast_vs_actual.png", bbox_inches="tight")
    plt.close(fig)


def plot_daily_total(df: pd.DataFrame, plt) -> None:
    """全 SKU 汇总日销量的实际 vs 预测（看整体水位是否跟得住）"""
    g = df.groupby("date").agg(actual=(config.TARGET, "sum"),
                               pred=("y_pred_model", "sum")).reset_index()
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(g["date"], g["actual"], label="actual total", color="#2c7fb8", lw=1.8)
    ax.plot(g["date"], g["pred"], label="forecast total", color="#e4572e", lw=1.6)
    ax.set_title("Test period: total daily units (all SKUs)", fontsize=10, loc="left")
    ax.set_ylabel("units", fontsize=9)
    ax.tick_params(labelsize=8)
    _format_date_axis(ax, interval=2)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "daily_total.png", bbox_inches="tight")
    plt.close(fig)


def plot_importance(plt) -> None:
    """特征重要性（收益 gain 口径）"""
    if not config.IMPORTANCE_FILE.exists():
        return
    imp = pd.read_csv(config.IMPORTANCE_FILE).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh(imp["feature"], imp["gain_share"], color="#3f7fbf")
    ax.set_xlabel("gain share", fontsize=9)
    ax.set_title("Top 15 features by gain", fontsize=10, loc="left")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "feature_importance.png", bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(metrics: dict, plt) -> None:
    """三种模型 × WMAPE 的对比条形图"""
    names = list(metrics.keys())
    values = [metrics[n]["WMAPE_%"] for n in names]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    bars = ax.bar([PLOT_LABEL.get(n, n) for n in names], values,
                  color=["#9aa5b1", "#c9b23f", "#2c7fb8"])
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("WMAPE (%) lower is better", fontsize=9)
    ax.set_title("Model comparison on test set", fontsize=10, loc="left")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "metric_comparison.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def run(make_plots: bool = True) -> dict:
    if not config.FORECAST_TEST_FILE.exists():
        from . import train
        train.run()
    df = pd.read_csv(config.FORECAST_TEST_FILE, encoding="utf-8-sig", parse_dates=["date"])

    result = evaluate_frame(df, only_uncensored=True)
    result["also_on_all_test_rows"] = evaluate_frame(df, only_uncensored=False)["metrics"]
    result["note"] = (
        "主指标只在实际未被断货截断的测试行上计算；also_on_all_test_rows 是把断货行也算进去的口径，"
        "会系统性低估实际需求，仅作对照。MAPE 剔除真实值为 0 的样本。"
    )

    # 分 SKU 明细
    sku_tbl = per_sku_table(df)
    sku_tbl.to_csv(config.RESULTS / "metrics_by_sku.csv", index=False, encoding="utf-8-sig")

    # 汇总打印
    print("[evaluate] 测试集评估（仅未断货行）")
    header = f"  {'模型':<16}{'MAE':>9}{'RMSE':>9}{'MAPE%':>9}{'WMAPE%':>9}"
    print(header)
    for name, m in result["metrics"].items():
        print(f"  {MLABEL.get(name, name):<16}{m['MAE']:>9.3f}{m['RMSE']:>9.3f}"
              f"{m['MAPE_%']:>9.2f}{m['WMAPE_%']:>9.2f}")
    print(f"  评估样本 {result['evaluated_rows']:,} 行；因断货剔除 "
          f"{result['censored_rows_excluded']:,} 行（占测试集 {result['censored_rate']:.2%}）")

    print("\n[by_pattern] 分需求模式的 WMAPE（%）：")
    for pattern, mm in result["by_pattern"].items():
        vals = " | ".join(f"{MLABEL.get(k, k)}={v['WMAPE_%']:.2f}" for k, v in mm.items())
        print(f"  {pattern:<9} {vals}")

    # 写 json
    config.METRICS_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[evaluate] → {config.METRICS_FILE.name} / metrics_by_sku.csv")

    if make_plots:
        try:
            plt = _setup_matplotlib()
            plot_forecast_curves(df, plt)
            plot_daily_total(df, plt)
            plot_importance(plt)
            plot_metric_comparison(result["metrics"], plt)
            print(f"[evaluate] → 图表已保存到 {config.FIGURES}")
        except Exception as exc:  # 画图失败不影响指标产出
            print(f"[evaluate] 绘图跳过：{exc}")
    return result


if __name__ == "__main__":
    run()
