"""
train.py —— 销量预测模型训练
================================================================
两个基线（必须打败它们，否则模型没有价值）：
    baseline_last_week  上周同一天销量（lag_7）
    baseline_ma_7       近 7 天移动平均（rolling_7_mean）

一个主模型：LightGBM（渐变提升树，对表格型零售数据最稳）。
    如果环境里没装 lightgbm，会自动降级：LightGBM → XGBoost → sklearn HistGradientBoosting。

三个刻意的选择：
    1) 用验证集 early stopping，测试集只在最后评估一次（不参与调参）；
    2) 训练集剔除断货截断行（y 被低估，标签不可信）；
    3) 预测值裁剪到 >=0（销量不可能为负）。

运行：
    python -m src.train
输出：
    results/forecast_test.csv    测试集逐行预测（评估与库存优化的输入）
    results/forecast_valid.csv   验证集逐行预测（用于估计预测误差 σ）
    results/feature_importance.csv
    results/model_lgbm.joblib    训练好的模型
    results/train_meta.json      训练元信息
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from . import config, features as feat_mod

# 基线模型：直接用特征表里已经算好的列当预测值
BASELINE_SPECS = {
    "baseline_last_week": "lag_7",
    "baseline_ma_7": "rolling_7_mean",
}

PROTECTED_COLS = {"units_sold", "demand_latent", "stockout_flag", "is_censored"}


def assert_no_leakage(feature_cols: list[str]) -> None:
    """保险丝：确保目标/真实需求/断货标记这类列绝不会混进特征"""
    leaked = PROTECTED_COLS.intersection(feature_cols)
    if leaked:
        raise ValueError(f"特征里混入了不可用的列（泄漏）：{sorted(leaked)}")


@dataclass
class TrainedModel:
    """统一封装不同后端的模型，下游只依赖 predict()"""

    name: str
    backend: str
    predict: Callable[[pd.DataFrame], np.ndarray]
    importance: pd.Series | None = None
    native: object | None = None
    best_iteration: int | None = None


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------
def prepare_matrices(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """按 split 切出 X/y，并剔除断货截断行"""
    feature_cols = feat_mod.FEATURE_COLS
    out = {}
    for split in ("train", "valid", "test"):
        part = df[df["split"] == split].copy()
        # 训练/验证集剔除截断行：这些行的 y 是"卖出去的"而不是"真实需求"
        fit_part = part[part["is_censored"] == 0] if split in ("train", "valid") else part
        out[split] = {
            "X": fit_part[feature_cols],
            "y": fit_part[config.TARGET].astype(float),
            "meta": fit_part,
            "n_all": int(len(part)),
            "n_censored_dropped": int(len(part) - len(fit_part)),
        }
    return out


# ---------------------------------------------------------------------------
# 各后端训练
# ---------------------------------------------------------------------------
def _fit_lightgbm(X_tr, y_tr, X_va, y_va, feature_cols) -> TrainedModel:
    import lightgbm as lgb

    params = dict(config.LGBM_PARAMS)
    params["seed"] = config.RANDOM_SEED
    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=feat_mod.CATEGORICAL_COLS,
                         free_raw_data=False)
    dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain, free_raw_data=False)
    booster = lgb.train(
        params, dtrain, num_boost_round=config.N_ROUNDS, valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False),
                   lgb.log_evaluation(0)],
    )
    importance = pd.Series(booster.feature_importance("gain"), index=feature_cols).sort_values(ascending=False)
    return TrainedModel(
        name="lightgbm", backend="lightgbm",
        predict=lambda X: booster.predict(X),
        importance=importance, native=booster,
        best_iteration=int(booster.best_iteration or booster.current_iteration()),
    )


def _fit_xgboost(X_tr, y_tr, X_va, y_va, feature_cols) -> TrainedModel:
    from xgboost import XGBRegressor

    model = XGBRegressor(
        n_estimators=config.N_ROUNDS, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.85, reg_lambda=1.0,
        objective="reg:squarederror", eval_metric="mae",
        early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
        random_state=config.RANDOM_SEED, n_jobs=4, tree_method="hist",
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    return TrainedModel(
        name="xgboost", backend="xgboost",
        predict=lambda X: model.predict(X),
        importance=importance, native=model,
        best_iteration=int(getattr(model, "best_iteration", 0) or 0),
    )


def _fit_sklearn(X_tr, y_tr, X_va, y_va, feature_cols) -> TrainedModel:
    """兜底方案：未装 lightgbm / xgboost 时用 sklearn 内置的直方图 GBDT"""
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=50, random_state=config.RANDOM_SEED,
    )
    model.fit(X_tr, y_tr)
    return TrainedModel(
        name="sklearn_hgb", backend="sklearn",
        predict=lambda X: model.predict(X),
        importance=None, native=model,
        best_iteration=int(getattr(model, "n_iter_", 0)),
    )


def train_main_model(X_tr, y_tr, X_va, y_va, feature_cols) -> TrainedModel:
    """按优先级选择后端：LightGBM → XGBoost → sklearn"""
    for fitter, pkg in ((_fit_lightgbm, "lightgbm"), (_fit_xgboost, "xgboost"), (_fit_sklearn, "sklearn")):
        try:
            model = fitter(X_tr, y_tr, X_va, y_va, feature_cols)
            print(f"[train] 主模型后端 = {model.backend}，best_iteration = {model.best_iteration}")
            return model
        except ImportError:
            print(f"[train] 未安装 {pkg}，尝试下一个后端…")
    raise RuntimeError("没有任何可用的 GBDT 后端，请安装 lightgbm 或 xgboost")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _safe_predict(model: TrainedModel, X: pd.DataFrame) -> np.ndarray:
    pred = np.asarray(model.predict(X), dtype=float)
    return np.clip(pred, 0.0, None)  # 销量不可能为负


def run() -> pd.DataFrame:
    """训练 + 推理 + 落盘，返回测试集预测明细"""
    # 1) 数据：特征表不存在时自动从原始数据一路生成
    if not config.FEATURE_FILE.exists():
        print("[train] 未找到特征表，自动执行 data_prep + features …")
        feat_mod.run()
    df = feat_mod.load_features()

    # 2) 切分
    data = prepare_matrices(df)
    tr, va, te = data["train"], data["valid"], data["test"]
    feature_cols = feat_mod.FEATURE_COLS
    assert_no_leakage(feature_cols)  # 双保险：特征列里绝不出现目标/真实需求

    n_censored_test = int(te["meta"]["is_censored"].sum())
    print(f"[train] 训练 {len(tr['X']):,} 行（剔除断货行 {tr['n_censored_dropped']}）| "
          f"验证 {len(va['X']):,} 行（剔除 {va['n_censored_dropped']}）| "
          f"测试 {len(te['X']):,} 行（其中断货行 {n_censored_test}，照常预测，评估时剔除）")

    # 3) 基线
    preds: dict[str, np.ndarray] = {}
    for name, col in BASELINE_SPECS.items():
        preds[name] = np.clip(te["X"][col].to_numpy(dtype=float), 0.0, None)

    # 4) 主模型
    model = train_main_model(tr["X"], tr["y"], va["X"], va["y"], feature_cols)
    preds["model"] = _safe_predict(model, te["X"])

    # 5) 输出：测试集逐行预测
    out_cols = ["sku_id", "date", "category", "sku_pattern",
                config.TARGET, "demand_latent", "stockout_flag", "is_censored",
                "promo_flag", "holiday_flag", "lag_7", "rolling_7_mean", "price"]
    fc_test = te["meta"][out_cols].copy()
    for name, p in preds.items():
        fc_test[f"y_pred_{name}"] = p
    fc_test.to_csv(config.FORECAST_TEST_FILE, index=False, encoding="utf-8-sig")

    # 6) 输出：验证集预测（只用未截断行，供 inventory 估计误差 σ）
    va_pred = _safe_predict(model, va["X"])
    fc_valid = va["meta"][["sku_id", "date", config.TARGET, "stockout_flag"]].copy()
    fc_valid["y_pred_model"] = va_pred
    fc_valid["residual"] = fc_valid[config.TARGET] - fc_valid["y_pred_model"]
    fc_valid.to_csv(config.FORECAST_VALID_FILE, index=False, encoding="utf-8-sig")

    # 7) 特征重要性
    if model.importance is not None:
        imp = model.importance.rename("gain").to_frame()
        imp.index.name = "feature"
        imp["gain_share"] = (imp["gain"] / imp["gain"].sum()).round(4)
        imp.to_csv(config.IMPORTANCE_FILE, encoding="utf-8-sig")
        print("[train] Top 10 重要特征：")
        print(imp.head(10).round(1).to_string())

    # 8) 保存模型
    try:
        import joblib
        joblib.dump(model.native, config.MODEL_FILE)
        # 记录相对路径而不是绝对路径：元信息会随结果一起提交，
        # 写绝对路径会把本地目录结构（用户名/盘符）泄漏到仓库里
        try:
            model_file = str(config.MODEL_FILE.relative_to(config.ROOT)).replace("\\", "/")
        except ValueError:
            model_file = config.MODEL_FILE.name
    except Exception as exc:  # 序列化失败不影响主流程
        model_file = None
        print(f"[train] 模型序列化失败（可忽略）：{exc}")

    # 9) 元信息
    split_info = {}
    if config.SPLIT_INFO_FILE.exists():
        split_info = json.loads(config.SPLIT_INFO_FILE.read_text(encoding="utf-8"))
    meta = {
        "model_name": model.name,
        "backend": model.backend,
        "best_iteration": model.best_iteration,
        "random_seed": config.RANDOM_SEED,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "n_train_rows": int(len(tr["X"])),
        "n_valid_rows": int(len(va["X"])),
        "n_test_rows": int(len(te["X"])),
        "censored_dropped_train": tr["n_censored_dropped"],
        "split": split_info,
        "model_file": model_file,
    }
    config.TRAIN_META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train] → {config.FORECAST_TEST_FILE.name} / {config.FORECAST_VALID_FILE.name} / "
          f"{config.TRAIN_META_FILE.name}")
    return fc_test


if __name__ == "__main__":
    run()
