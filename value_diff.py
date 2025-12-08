import pandas as pd
import argparse

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
    "label",
    "weight",
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

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--R", type=str, default="R_post.csv")
parser.add_argument("--Py", type=str, default="Py_post.csv")
args = parser.parse_args()

# load output.csv
R = pd.read_csv(args.R)
Py = pd.read_csv(args.Py)

R = R.round(3)
Py = Py.round(3)

Py = Py.rename(columns=coloum_shuffle)
R = R.rename(columns=coloum_shuffle)

# 对齐 merge（只保留两侧都有的数据）
df_merged = pd.merge(R, Py, on=["合约代码", "交易日期"], suffixes=("_R", "_Py"))

for col in factor_cols:
    if col in df_merged.columns:
        df_merged[f"{col}_diff"] = df_merged[f"{col}_R"] - df_merged[f"{col}_Py"]

# drop R and Py suffix columns
df_merged = df_merged.drop(
    columns=[f"{col}_R" for col in factor_cols] + [f"{col}_Py" for col in factor_cols]
)

# insert to the first row
sum_series = df_merged.filter(regex="diff$").abs().sum(axis=0)
df_final = pd.concat([pd.DataFrame([sum_series]), df_merged], ignore_index=True)

df_final.to_csv("RPy_value_diff.csv", index=False)
