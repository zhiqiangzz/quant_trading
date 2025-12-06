import pandas as pd

factor_cols = [
    "ROC_0",
    "RGap",
    "RangeLnHL",
    "CLV",
    "DLogVol",
    "ATRRelative_14",
    "DEVLogVol_7",
    "DEVEma_7",
    "Slope_7_14",
    "Spread_7_21",
    "HiEvent",
    "MACDHistNorm",
    "MACDHistCrossEvent",
    "BBandsPercentB",
    "BBandsWidthRelative",
    "BBandsBreakHigh",
    "ADX_14",
    "RSICenter",
    "RSIHiEvent",
    "HV_15_10",
    "d_HV",
    "ChaikinVolatility_10",
    "OBVFlow_7",
    "Mom_10",
    "RngMean_10",
    "DRngMean",
    "DomTop_20",
    "SkewTop_20",
    "DNetAtrTop_5_1",
    "DNetAtrTop_20_5",
    "LogReturnStockIndex_3_category",
    "LogReturnStockIndex_3_correlation",
    "Beta_60_40_category",
    "ExRet60_category",
    "Beta_60_40_correlation",
    "ExRet60_correlation",
    "BasisRate",
    "DaysToExpiry",
]
coloum_shuffle = {
    "r_oc_next": "ROCNext",
    "r_oc": "ROC_0",
    "r_gap": "RGap",
    "range_lnHL": "RangeLnHL",
    "CLV": "CLV",
    "d_logVol": "DLogVol",
    "ATR14_rel": "ATRRelative_14",
    "dev_logVol": "DEVLogVol_7",
    "dev_ema7": "DEVEma_7",
    "slope_ema7": "Slope_7_14",
    "spread_7_21": "Spread_7_21",
    "hi_event": "HiEvent",
    "MACD_hist_norm": "MACDHistNorm",
    "macd_cross_event": "MACDHistCrossEvent",
    "BB_pctB": "BBandsPercentB",
    "BB_width_rel": "BBandsWidthRelative",
    "bb_break_high": "BBandsBreakHigh",
    "ADX_14": "ADX_14",
    "RSI_center": "RSICenter",
    "rsi_hi_event": "RSIHiEvent",
    "HV_15": "HV_15_10",
    "d_HV_15": "d_HV",
    "Chaikin_Volatility": "ChaikinVolatility_10",
    "OBV_flow": "OBVFlow_7",
    "mom10": "Mom_10",
    "rng10_mean": "RngMean_10",
    "d_rng10_mean": "DRngMean",
    "Dom_top20": "DomTop_20",
    "Skew_top20": "SkewTop_20",
    "d1Net_ATR_top5": "DNetAtrTop_5_1",
    "d5Net_ATR_top20": "DNetAtrTop_20_5",
    "E_ret_lag3": "LogReturnStockIndex_3_correlation",
    "Beta_Equity": "Beta_60_40_correlation",
    "ExRet_Equity60": "ExRet60_correlation",
    "S_ret_lag3": "LogReturnStockIndex_3_category",
    "Beta_Sector": "Beta_60_40_category",
    "ExRet_Sector60": "ExRet60_category",
    "基差率": "BasisRate",
    "DaysToExpiry": "DaysToExpiry",
}

# load output.csv
R = pd.read_csv("R_contract_totrain.csv")
Py = pd.read_csv("Py_contract_totrain.csv")

R = R.round(3)
Py = Py.round(3)

# Set index so we can find intersection
R.set_index(["交易日期", "合约代码"], inplace=True)
Py.set_index(["交易日期", "合约代码"], inplace=True)

# Find common keys
common_index = R.index.intersection(Py.index)

# Filter and RESTORE index as columns
R_merge = R.loc[common_index].reset_index()
Py_merge = Py.loc[common_index].reset_index()

R_merge = R_merge.sort_values(["合约代码", "交易日期"])
Py_merge = Py_merge.sort_values(["合约代码", "交易日期"])

# Save with 合约代码 + 交易日期 columns included
R_merge.to_csv("R_merge.csv", index=False)
Py_merge.to_csv("Py_merge.csv", index=False)

R_merge = R_merge.rename(columns=coloum_shuffle)

# 对齐 merge（只保留两侧都有的数据）
df_merged = pd.merge(
    R_merge, Py_merge, on=["合约代码", "交易日期"], suffixes=("_R", "_Py")
)

for col in factor_cols:
    df_merged[f"{col}_diff"] = df_merged[f"{col}_R"] - df_merged[f"{col}_Py"]

# drop R and Py suffix columns
df_merged = df_merged.drop(
    columns=[f"{col}_R" for col in factor_cols] + [f"{col}_Py" for col in factor_cols]
)
df_merged.to_csv("R_Py_diff.csv", index=False)
