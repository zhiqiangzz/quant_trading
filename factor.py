import numpy as np
import pandas as pd
import utils
import argparse
import statsmodels.api as sm
import itertools
from functools import reduce
from itertools import chain
from collections import defaultdict
import re
from enum import Enum
from scipy.stats import spearmanr
import ta
import talib
from abc import ABC, abstractmethod


class BaseProcessor(ABC):
    def __init__(self):
        self.dfs = None

    def _make_result_df(self, factor_col):
        df = self.dfs[0]
        factor_col = [round(v, 3) for v in factor_col]
        return pd.DataFrame(
            {
                "交易日期": df["交易日期"],
                "合约代码": df["合约代码"],
                "factor_placeholder": factor_col,
            }
        )

    def run(self, dataframes: list[pd.DataFrame]):
        self.dfs = self._preprocess_all(dataframes)
        self._reset()
        return self._make_result_df(self.compute())

    def _unification_dates(self, dfs: list[pd.DataFrame]):
        common_dates = reduce(lambda x, y: x & y, [set(df["交易日期"]) for df in dfs])

        aligned_dfs = [
            df[df["交易日期"].isin(common_dates)].reset_index(drop=True) for df in dfs
        ]

        return aligned_dfs

    def _preprocess_all(self, dfs: list[pd.DataFrame]):
        dfs = [self.preprocess(df) for df in dfs]
        return self._unification_dates(dfs)

    def _canonicalize_factor_cal_df(self, factor_cal_df):
        # canonicalize column name
        rename_dict = {
            "名称": "品种名称",
            "日期": "交易日期",
            "date": "交易日期",
            "开盘价(元)": "开盘价",
            "收盘价(元)": "结算价",
            "基差(商品期货)\n[交易日期] 最新收盘日\n[单位] 元": "基差",
            "基差率(商品期货)\n[交易日期] 最新收盘日\n[单位] %": "基差率",
            "指标名称": "交易日期",
        }
        factor_cal_df.rename(columns=rename_dict, inplace=True)
        factor_cal_df.columns = factor_cal_df.columns.str.replace(
            r".*\(元/(吨|千克)\)$", "利润", regex=True
        )
        factor_cal_df.columns = factor_cal_df.columns.str.replace(
            r"中国:.*$", "现货价格", regex=True
        )

        factor_cal_df = utils.canonicalize_datetime_column(factor_cal_df)

        if "利润" in factor_cal_df.columns:
            factor_cal_df["利润"] = factor_cal_df["利润"].astype(float)
        if "现货价格" in factor_cal_df.columns:
            factor_cal_df["现货价格"] = factor_cal_df["现货价格"].astype(float)

        return factor_cal_df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._canonicalize_factor_cal_df(df)
        return df

    def _name_suffix(self):
        return ""

    def name(self):
        return self.__class__.__name__ + self._name_suffix()

    @abstractmethod
    def compute(self):
        pass

    def _reset(self):
        pass

    def is_aggregated(self):
        return False


class BaseProcessorWithParam(BaseProcessor):
    def __init__(self, params=7, *params_extra):
        self.params = [params] + list(params_extra)
        self.pop_idx = 0
        super().__init__()

    def _name_suffix(self):
        return f"_{'_'.join(str(k) for k in self.params)}"

    def _get_param(self):
        assert self.pop_idx < len(self.params), "No more parameters to pop"
        param = self.params[self.pop_idx]
        return param

    def _pop_param(self):
        param = self._get_param()
        self.pop_idx += 1
        return param

    def _reset(self):
        self.pop_idx = 0


class AggregateProcessorWithParam(BaseProcessorWithParam):
    def is_aggregated(self):
        return True


class AggregateProcessor(BaseProcessor):
    def is_aggregated(self):
        return True


class ROC(BaseProcessorWithParam):
    def compute(self):
        (df,) = self.dfs
        return np.log(df["收盘价"] / df["开盘价"].shift(self._pop_param()))


class ROCNext(BaseProcessor):
    def compute(self):
        (df,) = self.dfs
        return np.log(df["收盘价"] / df["开盘价"]).shift(-1)


# class LogReturn(BaseProcessor):
#     def __init__(self, lagging):
#         self.lagging = lagging
#         super().__init__()

#     def compute(self):
#         (df,) = self.dfs
#         return (np.log(df["结算价"] / df["结算价"].shift(self.lagging))).shift(1)


class RGap(BaseProcessor):
    def compute(self):
        (df,) = self.dfs
        return np.log(df["开盘价"] / df["收盘价"].shift(1))


class RangeLnHL(BaseProcessor):
    def compute(self):
        (df,) = self.dfs
        return np.log(df["最高价"] / df["最低价"])


class CLV(BaseProcessor):
    def compute(self):
        (df,) = self.dfs
        return np.where(
            df["最高价"] > df["最低价"],
            (df["收盘价"] - df["最低价"] - (df["最高价"] - df["收盘价"]))
            / (df["最高价"] - df["最低价"]),
            0,
        )


class DLogVol(BaseProcessor):
    def compute(self):
        (df,) = self.dfs
        return np.log1p(df["成交量"]) - np.log(df["成交量"].shift(1))


class ATR(BaseProcessorWithParam):
    def compute(self):
        (df,) = self.dfs
        atr = ta.volatility.average_true_range(
            df["最高价"], df["最低价"], df["收盘价"], window=self._pop_param()
        )

        ATRx = np.maximum(atr, 1e-8)
        return ATRx


class ATRRelative(BaseProcessorWithParam):
    def compute(self):
        (df,) = self.dfs
        ATR_rel = df[f"ATR_{self._pop_param()}"] / np.maximum(
            df["收盘价"].shift(1), 1e-8
        )
        return ATR_rel


class DEVLogVol(BaseProcessorWithParam):
    def compute(self):
        (df,) = self.dfs
        logVol = np.log1p(df["成交量"])
        return logVol - logVol.ewm(span=self._pop_param(), adjust=False).mean()


class EMA(BaseProcessorWithParam):
    def compute(self):
        (df,) = self.dfs
        return df["收盘价"].ewm(span=self._pop_param(), adjust=False).mean()


class DEVEma(BaseProcessorWithParam):
    def compute(self):
        (df,) = self.dfs
        return (df["收盘价"] - df[f"EMA_{self._pop_param()}"]) / df["ATR_14"]


class Slope(BaseProcessorWithParam):
    def compute(self):
        (df,) = self.dfs
        return (
            df[f"EMA_{self._get_param()}"] - df[f"EMA_{self._pop_param()}"].shift(1)
        ) / df[f"ATR_14"]


class Spread(BaseProcessorWithParam):
    def compute(self):
        (df,) = self.dfs
        return (df[f"EMA_{self._pop_param()}"] - df[f"EMA_{self._pop_param()}"]) / df[
            f"ATR_14"
        ]


class DEVEmaQuantile(BaseProcessorWithParam):
    def compute(self):
        (df,) = self.dfs
        return (
            df[f"DEVEma_7"]
            .rolling(window=self._pop_param(), min_periods=1)
            .apply(lambda x: np.quantile(x, self._get_param() / 100))
            .shift(1)
        )


class NumEffectiveDays(BaseProcessorWithParam):
    def compute(self):
        (df,) = self.dfs
        num_effective_days = (
            df[f"DEVEma_7"]
            .rolling(window=self._get_param(), min_periods=1)
            .apply(lambda x: np.sum(~np.isnan(x)) >= self._get_param())
            .shift(1)
        )
        return num_effective_days


class HiEvent(BaseProcessor):
    def compute(self):
        (df,) = self.dfs
        num_effective_days = df["NumEffectiveDays_60"].fillna(False).astype(bool)
        return num_effective_days & (df[f"DEVEma_7"] > df[f"DEVEmaQuantile_60_80"])


class LoEvent(BaseProcessor):
    def compute(self):
        (df,) = self.dfs
        num_effective_days = df["NumEffectiveDays_60"].fillna(False).astype(bool)
        return num_effective_days & (df[f"DEVEma_7"] < df[f"DEVEmaQuantile_60_20"])


#   # MACD
#   macd <- MACD(contract_data$收盘价, nFast = 12, nSlow = 26, nSig = 9, maType = "EMA")
#   contract_data <- contract_data %>%
#     mutate(
#       MACD_line   = macd[, "macd"],
#       MACD_signal = macd[, "signal"],
#       MACD_hist   = MACD_line - MACD_signal,
#       MACD_hist_norm = MACD_hist / pmax(ATR14, 1e-8),                   # 连续量
#       macd_cross_event = as.integer(sign(MACD_hist) != sign(lag(MACD_hist)) &  # 事件
#                                       !is.na(MACD_hist) & !is.na(lag(MACD_hist)))
#     )


def DEMA(span):
    def func(_, factor_var):
        # 第一次 EMA
        ema1 = factor_var["结算价"].ewm(span=span, adjust=False).mean()
        # 第二次 EMA（对 ema1 再做一次 EMA）
        ema2 = ema1.ewm(span=span, adjust=False).mean()
        # DEMA 公式
        return 2 * ema1 - ema2

    return func


def Beta():
    def func(var, factor_var):
        var["ret"] = np.log(var["结算价"] / var["结算价"].shift(1))
        factor_var["mret"] = np.log(
            factor_var["结算价"] / factor_var["结算价"].shift(1)
        )

        df = pd.merge(
            var[["交易日期", "ret"]], factor_var[["交易日期", "mret"]], on="交易日期"
        ).dropna()

        X = sm.add_constant(df["mret"])
        model = sm.OLS(df["ret"], X).fit()
        beta = model.params["mret"]

        return df["ret"] - beta * df["mret"]

    return func


def BasisReturn():
    def func(var, factor_var):
        return np.log(factor_var["结算价"] / factor_var["结算价"].shift(1)).shift(1)

    return func


def RegionRelative():
    def func(_, factor_var):
        return (
            (factor_var["最高价"] - factor_var["最低价"])
            / factor_var["结算价"].shift(1)
        ).shift(1)

    return func


def DirectionalStrength():
    def func(_, factor_var):
        return (
            (factor_var["收盘价"] - factor_var["开盘价"])
            / (factor_var["最高价"] - factor_var["最低价"])
        ).shift(1)

    return func


def IntradayReturns():
    def func(_, factor_var):
        return np.log(factor_var["收盘价"] / factor_var["开盘价"]).shift(1)

    return func


def TradingVolumeGrowthRate():
    def func(_, factor_var):
        return np.log(factor_var["成交量"] / factor_var["成交量"].shift(1)).shift(1)

    return func


def OpenInterestGrowthRate():
    def func(_, factor_var):
        return np.log(factor_var["持仓量"] / factor_var["持仓量"].shift(1)).shift(1)

    return func


def KdaysLogMomentum(days: int):
    def func(_, factor_var):
        return (
            np.log(factor_var["收盘价"]) - np.log(factor_var["收盘价"].shift(days))
        ).shift(1)

    return func


def ZScore(min_effective_days: int, max_effective_days: int):
    def func(_, factor_var):
        KMeans = pd.Series(np.zeros(len(factor_var)), name="KMeans")
        KStd = pd.Series(np.zeros(len(factor_var)), name="KStd")
        for i in range(min_effective_days, max_effective_days):
            KMeans[i] = factor_var["结算价"].iloc[:i].mean()
            KStd[i] = factor_var["结算价"].iloc[:i].std()
        for i in range(max_effective_days, len(factor_var)):
            KMeans[i] = factor_var["结算价"].iloc[i - max_effective_days : i].mean()
            KStd[i] = factor_var["结算价"].iloc[i - max_effective_days : i].std()
        return ((factor_var["结算价"] - KMeans.shift(1)) / KStd.shift(1)).shift(1)

    return func


def KdaysRegionRelativeMeanAndVary(kdays: int):
    def func(_, factor_var):
        HLDiff = (factor_var["最高价"] - factor_var["最低价"]) / factor_var[
            "结算价"
        ].shift(1)
        # kdays mean
        KMeans = pd.Series(np.zeros(len(HLDiff)), name="KMeans")
        for i in range(0, kdays):
            KMeans[i] = HLDiff.iloc[:i].mean()
        for i in range(kdays, len(HLDiff)):
            KMeans[i] = HLDiff.iloc[i - kdays : i].mean()
        return KMeans.shift(1)

    return func


def KdaysGrowthRateTradingVolume(kdays: int):
    def func(_, factor_var):
        return ((factor_var["成交量"] / factor_var["成交量"].shift(kdays)) - 1).shift(1)

    return func


def KdaysOpenInterest(kdays: int):
    def func(_, factor_var):
        return ((factor_var["持仓量"] / factor_var["持仓量"].shift(kdays)) - 1).shift(1)

    return func


def KdaysTradingVolumeRelativeOffset(kdays: int):
    def func(_, factor_var):
        EMAK = np.log(factor_var["成交量"]).ewm(span=kdays, adjust=False).mean()
        return (np.log(factor_var["成交量"]) - EMAK.shift(1)).shift(1)

    return func


class Basis(BaseProcessor):
    def compute(self):
        _, basis_df = self.dfs
        return basis_df["基差"].shift(1)


class BasisRate(BaseProcessor):
    def compute(self):
        _, basis_df = self.dfs
        return basis_df["基差率"].shift(1)


def ProfitMargin():
    def func(profit_var, spot_price_var):
        return (profit_var["利润"] / spot_price_var["现货价格"]).shift(1)

    return func


def DailyProfit(lagging: int = 1):
    def func(_, factor_var):
        return factor_var["利润"].shift(lagging)

    return func


def Production(lagging: int = 1):
    def func(_, factor_var):
        return factor_var["产量"].shift(lagging)

    return func


def ProductionGrowthRate(lagging: int = 1):
    def func(_, factor_var):
        return (factor_var["产量"] / factor_var["产量"].shift(lagging)).shift(1)

    return func


def MACDHelper(series, n):
    if series.isna().all() or len(series) < n + series.first_valid_index():
        return pd.Series(np.nan, index=series.index)

    ema = series[series.first_valid_index() :].copy()
    ema.iloc[: n - 1] = np.nan
    ema.iloc[n - 1] = series.iloc[:n].mean()
    ema.iloc[n - 1 :] = ema.ewm(span=n, adjust=False).mean().iloc[n - 1 :]
    ema = ema.reindex(series.index).fillna(np.nan)
    return ema


class MACDLine(BaseProcessorWithParam):
    def compute(self):
        (df,) = self.dfs
        input_series = df["收盘价"]

        return 100 * (
            MACDHelper(input_series, self._pop_param())
            / MACDHelper(input_series, self._pop_param())
            - 1
        )


class MACDSignal(BaseProcessorWithParam):
    def compute(self):
        (df,) = self.dfs

        return MACDHelper(df["MACDLine_12_26"], self._pop_param())


class BBandsUpper(BaseProcessorWithParam):
    def compute(self):
        (df,) = self.dfs
        bb = talib.BBANDS(
            df["收盘价"], timeperiod=self._pop_param(), nbdevup=2, nbdevdn=2, matype=0
        )
        return bb[0]


class BBandsMiddle(BaseProcessorWithParam):
    def compute(self):
        (df,) = self.dfs
        bb = talib.BBANDS(
            df["收盘价"], timeperiod=self._pop_param(), nbdevup=2, nbdevdn=2, matype=0
        )
        return bb[1]


class BBandsLower(BaseProcessorWithParam):
    def compute(self):
        (df,) = self.dfs
        bb = talib.BBANDS(
            df["收盘价"], timeperiod=self._pop_param(), nbdevup=2, nbdevdn=2, matype=0
        )
        return bb[2]


class BBandsWidth(BaseProcessor):
    def compute(self):
        (df,) = self.dfs
        return df["BBandsUpper_20"] - df["BBandsLower_20"]


class BBandsPercentB(BaseProcessor):
    def compute(self):
        (df,) = self.dfs
        return (df["收盘价"] - df["BBandsLower_20"]) / (df["BBandsWidth"])


class BBandsWidthRelative(BaseProcessor):
    def compute(self):
        (df,) = self.dfs
        return df["BBandsWidth"] / np.maximum(df["BBandsMiddle_20"], 1e-8)


class BBandsBreakHigh(BaseProcessor):
    def compute(self):
        (df,) = self.dfs
        return df["BBandsPercentB"] > 0.8


class BBandsBreakLow(BaseProcessor):
    def compute(self):
        (df,) = self.dfs
        return df["BBandsPercentB"] < 0.2


class ADX(BaseProcessorWithParam):
    def compute(self):
        (df,) = self.dfs
        adx = talib.ADX(
            df["最高价"], df["最低价"], df["收盘价"], timeperiod=self._pop_param()
        )
        return adx


class RSI(BaseProcessorWithParam):
    def compute(self):
        (df,) = self.dfs
        rsi = talib.RSI(df["收盘价"], timeperiod=self._pop_param())
        return rsi


class RSICenter(BaseProcessor):
    def compute(self):
        (df,) = self.dfs
        return df["RSI_14"] - 50


class RSIHiEvent(BaseProcessor):
    def compute(self):
        (df,) = self.dfs
        return df["RSI_14"] > 70


class RSILoEvent(BaseProcessor):
    def compute(self):
        (df,) = self.dfs
        return df["RSI_14"] < 30


class HV(BaseProcessorWithParam):
    def compute(self):
        (df,) = self.dfs
        return (
            df["收盘价"]
            .apply(np.log)
            .diff()
            .rolling(window=self._pop_param(), min_periods=self._pop_param())
            .std()
        )


class d_HV(BaseProcessor):
    def compute(self):
        (df,) = self.dfs
        return df["HV_15_10"].diff()


def ChaikinVolatilityHelper(high, low, period=10):
    hl_range = high - low
    ema_hl = hl_range.ewm(span=period, adjust=False).mean()
    chaikin_vol = (ema_hl - ema_hl.shift(period)) / ema_hl.shift(period)
    return chaikin_vol


class ChaikinVolatility(BaseProcessorWithParam):
    def compute(self):
        (df,) = self.dfs
        return ChaikinVolatilityHelper(
            df["最高价"],
            df["最低价"],
            self._pop_param(),
        )


class OBVFlow(BaseProcessorWithParam):
    def compute(self):
        (df,) = self.dfs
        obv = ta.volume.on_balance_volume(df["收盘价"], df["成交量"])
        vol_ema7 = df["成交量"].ewm(span=self._pop_param(), adjust=False).mean()
        return (obv - obv.shift(1)) / np.maximum(vol_ema7, 1e-8)


class Mom(BaseProcessorWithParam):
    def compute(self):
        (df,) = self.dfs
        return np.log(df["收盘价"]) - np.log(df["收盘价"].shift(self._pop_param()))


class RngRel(BaseProcessor):
    def compute(self):
        (df,) = self.dfs
        return (df["最高价"] - df["最低价"]) / df["收盘价"].shift(1)


class RngMean(BaseProcessorWithParam):
    def compute(self):
        (df,) = self.dfs
        return (
            df["RngRel"]
            .rolling(window=self._get_param(), min_periods=self._pop_param())
            .mean()
        )


class DRngMean(BaseProcessor):
    def compute(self):
        (df,) = self.dfs
        return df["RngMean_10"].diff()


class UpDownNext(BaseProcessor):
    def compute(self):
        (df,) = self.dfs
        return np.where(df["ROCNext"] > 0, 1, 0)


class RelativeToSettle(AggregateProcessor):
    def compute(self):
        (df,) = self.dfs
        return (
            df.groupby("交易日期")
            .apply(
                lambda x: (
                    x["结算价"] - x.loc[x["IsMain"] == 1, "结算价"].iloc[0]
                    if (x["IsMain"] == 1).any()
                    else np.nan
                )
            )
            .reset_index(level=0, drop=True)
        )


class DiffAbsAtr(AggregateProcessor):
    def compute(self):
        (df,) = self.dfs
        return np.abs(df["RelativeToSettle"] / np.maximum(df["ATR_14"], 1e-8))


class LastDayClosePrice(BaseProcessor):
    def compute(self):
        (df,) = self.dfs
        return df["收盘价"].shift(1)


class OISum(AggregateProcessor):
    def compute(self):
        (df,) = self.dfs
        OISum = (
            df.groupby("交易日期").apply(lambda x: x["持仓量"].sum()).rename("OISum")
        )

        return df.merge(OISum, on="交易日期", how="left")["OISum"]


class TopNet(AggregateProcessorWithParam):
    def compute(self):
        (df, net_pos_df) = self.dfs
        topN = self._pop_param()
        net_pos_df["TopNet"] = (
            net_pos_df[f"long_position_top{topN}"]
            - net_pos_df[f"short_position_top{topN}"]
        )
        return df.merge(net_pos_df, on="交易日期", how="left")["TopNet"]


class TopTotal(AggregateProcessorWithParam):
    def compute(self):
        (df, net_pos_df) = self.dfs
        topN = self._pop_param()
        net_pos_df["TopTotal"] = (
            net_pos_df[f"long_position_top{topN}"]
            + net_pos_df[f"short_position_top{topN}"]
        )
        return df.merge(net_pos_df, on="交易日期", how="left")["TopTotal"]


class DomTop(AggregateProcessorWithParam):
    def compute(self):
        df, _ = self.dfs
        topN = self._pop_param()
        return np.where(
            (df["OISum"] <= 0) | df["OISum"].isna(),
            np.nan,
            df[f"TopTotal_{topN}"] / (2 * df["OISum"]),
        )


class SkewTop(AggregateProcessorWithParam):
    def compute(self):
        df, _ = self.dfs
        topN = self._pop_param()
        return np.where(
            (df["OISum"] <= 0) | df["OISum"].isna(),
            np.nan,
            df[f"TopNet_{topN}"] / df[f"TopTotal_{topN}"],
        )


# TR = df.apply(
#     lambda row: max(
#         [
#             row["最高价"] - row["最低价"],
#             abs(row["最高价"] - row["收盘价"].shift(1)),
#             abs(row["最低价"] - row["收盘价"].shift(1)),
#         ],
#         default=np.nan,
#     ),
#     axis=1,
# )

# def weighted_mean(group):
#     w = group["成交量"]
#     v = group["TR"]
#     if w.isna().all() or w.sum() <= 0:
#         return v.mean()
#     else:
#         return np.average(v, weights=w)

# TR_agg = TR.groupby("交易日期").apply(lambda x: weighted_mean(x))
# ATR14 = TR_agg.rolling(window=14, min_periods=1).mean()
