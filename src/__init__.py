"""src 包：AI 辅助电商销量预测与库存优化。

模块职责：
    data_generator  生成模拟电商日粒度销量数据（含断货截断）
    data_prep       清洗、补全、质量校验
    features        滞后 / 滑动窗口 / 促销 / 节假日特征 + 时间切分
    train           基线模型与 LightGBM/XGBoost 主模型训练
    evaluate        MAE / RMSE / MAPE / WMAPE 评估
    inventory       安全库存 / 再订货点 / 建议补货量
    config          全局配置
"""

__version__ = "0.1.0"
