"""
config.py —— 全局配置（路径 / 随机种子 / 数据规模 / 模型参数 / 服务水平）
所有可调参数集中在这里，改一处即可全局生效；以后接真实数据时主要改这个文件 + data_prep。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# 一、路径：全部基于项目根目录推导，保证在任何机器上 cd 到项目根都能跑
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

for _p in (DATA_RAW, DATA_PROCESSED, RESULTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

RAW_FILE = DATA_RAW / "sales_daily.csv"          # 原始模拟数据（只读，不覆盖式改写）
CLEAN_FILE = DATA_PROCESSED / "sales_clean.csv"  # 清洗后数据
FEATURE_FILE = DATA_PROCESSED / "features.csv"   # 特征工程 + 时间切分结果
SPLIT_INFO_FILE = DATA_PROCESSED / "split_info.json"

FORECAST_TEST_FILE = RESULTS / "forecast_test.csv"
FORECAST_VALID_FILE = RESULTS / "forecast_valid.csv"
TRAIN_META_FILE = RESULTS / "train_meta.json"
METRICS_FILE = RESULTS / "model_metrics.json"
IMPORTANCE_FILE = RESULTS / "feature_importance.csv"
REPLENISHMENT_FILE = RESULTS / "replenishment_table.csv"
MODEL_FILE = RESULTS / "model_lgbm.joblib"

# ---------------------------------------------------------------------------
# 二、可复现性：所有随机过程都从这一个种子派生
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# 三、模拟数据规模与口径
# ---------------------------------------------------------------------------
N_DAYS = 180                      # 观察窗口：截止日前 180 天
ANCHOR_DATE = date(2026, 10, 23)  # 数据截止日（固定值 → 结果可复现；接真实数据时忽略）
N_SKUS = 40                       # SKU 总数（4 种模式各 10 个）

TARGET = "units_sold"             # 预测目标：日销量

# SKU 需求模式：按顺序轮转分配，保证每种模式都有足够样本
PATTERNS = ("stable", "promo", "seasonal", "longtail")

# 品类 → 价格区间（元），用于让 price 有业务量级差异
CATEGORIES = {
    "美妆个护": (69.0, 299.0),
    "家居日用": (19.9, 159.0),
    "数码配件": (39.0, 399.0),
    "休闲食品": (9.9, 79.0),
    "服饰鞋包": (49.0, 459.0),
}

# 法定节假日区间（含首尾）。真实项目请改用 chinese_calendar 等库自动生成
HOLIDAY_RANGES = [
    (date(2025, 5, 1), date(2025, 5, 5)),
    (date(2025, 5, 31), date(2025, 6, 2)),
    (date(2025, 10, 1), date(2025, 10, 8)),
    (date(2026, 1, 1), date(2026, 1, 3)),
    (date(2026, 2, 15), date(2026, 2, 23)),
    (date(2026, 4, 4), date(2026, 4, 6)),
    (date(2026, 5, 1), date(2026, 5, 5)),
    (date(2026, 6, 19), date(2026, 6, 21)),
    (date(2026, 9, 25), date(2026, 9, 27)),
    (date(2026, 10, 1), date(2026, 10, 8)),
]
# 节假日前后一天的“出行/囤货”抬升（仅用于生成更真实的节假日效应）
HOLIDAY_LIFT_WINDOW = 1

# ---------------------------------------------------------------------------
# 四、时间切分：严格按时间顺序，绝不随机切分
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15
WARMUP_DAYS = 14  # 前 14 天用于构造 lag_14 / rolling_30，不参与训练与评估
ROLLING_MIN_PERIODS = 14  # rolling 特征的最少观测数（不足则视为 warmup）

# ---------------------------------------------------------------------------
# 五、模型参数
# ---------------------------------------------------------------------------
LGBM_PARAMS = {
    "objective": "regression",   # 可改 "regression_l1" / "huber" 抗促销尖峰
    "metric": "mae",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "num_threads": 4,
    "deterministic": True,
    "force_row_wise": True,
}
N_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 80

# ---------------------------------------------------------------------------
# 六、库存策略参数
# ---------------------------------------------------------------------------
SERVICE_LEVEL = 0.95        # 目标服务水平 → z 值（95% 对应 1.645）
REVIEW_PERIOD_DAYS = 7      # 补货复核周期（天）：每 7 天下一次单
SIGMA_SHRINK_WEIGHT = 0.4   # 单 SKU 误差标准差向整体池化值收缩的权重（0=完全用自身，1=完全用全局）
ORDER_MULTIPLE = 1          # 最小起订量/装箱数的整数倍
DEMAND_WINDOW_FOR_RATE = 28 # 用测试期最后 28 天的平均预测，作为日需求速率 mu_d
