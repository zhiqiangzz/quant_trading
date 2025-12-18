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
import model
from abc import ABC, abstractmethod
from factor import *
import global_var

CONFIG = {
    "index": {
        "correlation": {
            "files": {
                "碳酸锂": ["商品关联指数/电池931719.CSI.xlsx"],
                "工业硅": ["商品关联指数/光伏931151.CSI.xlsx"],
                "沪镍": ["商品关联指数/有色801050.SI.xlsx"],
                "沪锡": ["商品关联指数/有色801050.SI.xlsx"],
                "沪铜": ["商品关联指数/有色801050.SI.xlsx"],
                "沪银": ["商品关联指数/贵金属801053.SI.xlsx"],
                "螺纹钢": ["商品关联指数/钢铁801040.SI.xlsx"],
                "氧化铝": ["商品关联指数/有色801050.SI.xlsx"],
                "橡胶": ["商品关联指数/化工801030.SI.xlsx"],
                "纯碱": ["商品关联指数/化工801030.SI.xlsx"],
                "烧碱": ["商品关联指数/化工801030.SI.xlsx"],
                "玻璃": [
                    "商品关联指数/建筑材料801710.SI.xlsx",
                ],
                "甲醇": ["商品关联指数/化工801030.SI.xlsx"],
                "精对苯二甲酸": [
                    "商品关联指数/化学纤维801032.SI.xlsx",
                ],
                "铁矿石": [
                    "商品关联指数/钢铁801040.SI.xlsx",
                ],
                "豆粕": ["商品关联指数/饲料801014.SI.xlsx"],
                "棕榈油": [
                    "商品关联指数/农产品801012.SI.xlsx",
                ],
                "焦煤": [
                    "商品关联指数/煤炭000820.CSI.xlsx",
                ],
            },
            "factors": [
                # LogReturn(1),
                LogReturnStockIndex([0], "correlation"),
                LogReturnStockIndex([3], "correlation"),
                Beta([60, 40], "correlation"),
                ExRet60(name_suffix="correlation"),
                # LogReturn(5),
                # Beta(),
            ],
        },
        "category": {
            "files": {
                "碳酸锂": ["商品品类指数/化工CIFI.WI.xlsx"],
                # "工业硅": [
                #     "商品品类指数/南华新材料NHNMI.NHF.xlsx",
                # ],
                # "沪镍": ["商品品类指数/有色NFFI.WI.xlsx"],
                # "沪锡": ["商品品类指数/有色NFFI.WI.xlsx"],
                # "沪铜": ["商品品类指数/有色NFFI.WI.xlsx"],
                # "沪银": ["商品品类指数/贵金属NMFI.WI.xlsx"],
                # "螺纹钢": ["商品品类指数/煤焦钢矿JJRI.WI.xlsx"],
                # "氧化铝": ["商品品类指数/有色NFFI.WI.xlsx"],
                # "橡胶": ["商品品类指数/软商品SOFI.WI.xlsx"],
                # "纯碱": ["商品品类指数/化工CIFI.WI.xlsx"],
                # "烧碱": ["商品品类指数/化工CIFI.WI.xlsx"],
                # "玻璃": [
                #     "商品品类指数/非金属建材NMBM.WI.xlsx",
                # ],
                # "甲醇": ["商品品类指数/化工CIFI.WI.xlsx"],
                # "精对苯二甲酸": [
                #     "商品品类指数/化工CIFI.WI.xlsx",
                # ],
                # "铁矿石": [
                #     "商品品类指数/南华黑色原材料NHFMI.NHF.xlsx",
                # ],
                # "豆粕": ["商品品类指数/油脂油料OOFI.WI.xlsx"],
                # "棕榈油": [
                #     "商品品类指数/油脂油料OOFI.WI.xlsx",
                # ],
                # "焦煤": [
                #     "商品品类指数/煤焦钢矿JJRI.WI.xlsx",
                # ],
            },
            "factors": [
                # LogReturn(1),
                LogReturnStockIndex([0], "category"),
                LogReturnStockIndex([3], "category"),
                Beta([60, 40], "category"),
                ExRet60(name_suffix="category"),
                # LogReturn(5),
                # Beta(),
            ],
        },
    },
    "fundamentals": {
        # "profit": {
        #     "files": {
        #         "碳酸锂": [
        #             (
        #                 "利润（日）/碳酸锂理论日度利润-毛利润.xlsx",
        #                 "现货价格/中国_现货领先价格_碳酸锂.xls",
        #             )
        #         ],
        #         "工业硅": [
        #             (
        #                 "利润（日）/工业硅理论日度成本利润-毛利润.xlsx",
        #                 "现货价格/中国_现货领先价格_工业硅.xls",
        #             )
        #         ],
        #         "沪镍": [
        #             (
        #                 "利润（日）/镍理论日度成本利润-毛利润.xlsx",
        #                 "现货价格/中国_现货领先价格_镍.xls",
        #             )
        #         ],
        #         "沪锡": [
        #             (
        #                 "利润（日）/锡理论成本利润-毛利润.xlsx",
        #                 "现货价格/中国_现货领先价格_锡.xls",
        #             )
        #         ],
        #         "沪铜": [
        #             (
        #                 "利润（日）/铜理论日度利润-毛利润.xlsx",
        #                 "现货价格/中国_现货领先价格_阴极铜.xls",
        #             )
        #         ],
        #         "沪银": [
        #             (
        #                 "利润（日）/银理论成本利润-毛利润.xlsx",
        #                 "现货价格/中国_现货领先价格_银.xls",
        #             )
        #         ],
        #         "螺纹钢": [
        #             (
        #                 "利润（日）/螺纹钢理论日度利润-毛利润.xlsx",
        #                 "现货价格/中国_现货领先价格_螺纹钢.xls",
        #             )
        #         ],
        #         "氧化铝": [
        #             (
        #                 "利润（日）/氧化铝理论日度利润-毛利润.xlsx",
        #                 "现货价格/中国_现货领先价格_氧化铝.xls",
        #             )
        #         ],
        #         "橡胶": [
        #             (
        #                 "利润（日）/橡胶理论日度利润-毛利润.xlsx",
        #                 "现货价格/中国_现货领先价格_橡胶.xls",
        #             )
        #         ],
        #         "纯碱": [
        #             (
        #                 "利润（日）/纯碱理论日度利润-毛利润.xlsx",
        #                 "现货价格/中国_现货领先价格_纯碱.xls",
        #             )
        #         ],
        #         # "烧碱":[("利润（日）/烧碱理论日度利润-毛利润.xlsx","现货价格/中国_现货领先价格_烧碱.xls")],
        #         "玻璃": [
        #             (
        #                 "利润（日）/玻璃理论日度利润-毛利润.xlsx",
        #                 "现货价格/中国_现货领先价格_玻璃.xls",
        #             )
        #         ],
        #         "甲醇": [
        #             (
        #                 "利润（日）/甲醇理论日度利润-毛利润.xlsx",
        #                 "现货价格/中国_现货领先价格_甲醇.xls",
        #             )
        #         ],
        #         "精对苯二甲酸": [
        #             (
        #                 "PTA理论日度利润（日）-毛利润.xlsx",
        #                 "现货价格/中国_现货领先价格_精对苯二甲酸(PTA).xls",
        #             )
        #         ],
        #         "铁矿石": [
        #             (
        #                 "利润（日）/铁矿石理论成本利润-毛利润.xlsx",
        #                 "现货价格/中国_现货领先价格_铁矿石.xls",
        #             )
        #         ],
        #         "豆粕": [
        #             (
        #                 "利润（日）/豆粕理论日度利润-毛利润.xlsx",
        #                 "现货价格/中国_现货领先价格_豆粕.xls",
        #             )
        #         ],
        #         "棕榈油": [
        #             (
        #                 "利润（日）/棕榈油理论日度利润-毛利润.xlsx",
        #                 "现货价格/中国_现货领先价格_棕榈油.xls",
        #             )
        #         ],
        #         "焦煤": [
        #             (
        #                 "利润（日）/炼焦煤理论日度毛利润-毛利润.xlsx",
        #                 "现货价格/中国_现货领先价格_焦煤.xls",
        #             )
        #         ],
        #     },
        #     "factors": {
        #         # "ProfitMargin": (ProfitMargin()),
        #         # "DailyProfitLag1": (DailyProfit(1)),
        #         # "DailyProfitLag5": (DailyProfit(5)),
        #         # "DailyProfitLag15": (DailyProfit(15)),
        #     },
        # },
        # "production": {
        #     "files": {
        #         "碳酸锂": ["产量/碳酸锂产量.xls"],
        #         "工业硅": ["产量/工业硅产量.xls"],
        #         "沪镍": ["产量/镍产量.xls"],
        #         "沪锡": ["产量/锡产量.xls"],
        #         "沪铜": ["产量/铜产量.xls"],
        #         "沪银": ["产量/银产量.xls"],
        #         "螺纹钢": ["产量/螺纹钢产量.xls"],
        #         "氧化铝": ["产量/氧化铝产量.xls"],
        #         "橡胶": ["产量/橡胶产量.xls"],
        #         "纯碱": ["产量/纯碱总产量.xls"],
        #         # "烧碱":["产量/烧碱产量.xls"],
        #         "玻璃": ["产量/玻璃产量.xls"],
        #         "甲醇": ["产量/甲醇产量.xls"],
        #         "精对苯二甲酸": ["产量/PTA产量.xls"],
        #         # "铁矿石":["产量/铁矿石统计局产量.xls"],
        #         "豆粕": ["产量/豆粕产量.xls"],
        #         # "棕榈油":["产量/棕榈油产量.xls"],
        #         # "焦煤":["产量/焦煤产量.xls"],
        #     },
        #     "factors": {
        #         # "ProductionLag1": (Production(1)),
        #     },
        # },
        "net_positions": {
            "files": {
                "碳酸锂": ["机构净持仓/[碳酸锂 LC]_净持仓合计_wide.xlsx"],
            },
            "factors": [
                TopNet([20]),
                TopTotal([20]),
                TopNet([5]),
                TopTotal([5]),
                DomTop([20]),
                SkewTop([20]),
                DNetAtrTop([5, 1]),
                DNetAtrTop([20, 5]),
            ],
        },
    },
    "basic": {
        "basic": {
            "files": {
                "碳酸锂": ["基差/碳酸锂.xlsx"],
                # "工业硅": ["基差/工业硅.xlsx"],
                # "沪镍": ["基差/镍.xlsx"],
                # "沪锡": ["基差/沪锡.xlsx"],
                # "沪铜": ["基差/沪铜.xlsx"],
                # "沪银": ["基差/白银.xlsx"],
                # "螺纹钢": ["基差/螺纹.xlsx"],
                # "氧化铝": ["基差/氧化铝.xlsx"],
                # "橡胶": ["基差/橡胶.xlsx"],
                # "纯碱": ["基差/纯碱.xlsx"],
                # "烧碱": ["基差/烧碱.xlsx"],
                # "玻璃": ["基差/玻璃.xlsx"],
                # "甲醇": ["基差/甲醇.xlsx"],
                # "精对苯二甲酸": ["基差/PTA.xlsx"],
                # "铁矿石": ["基差/铁矿石.xlsx"],
                # "豆粕": ["基差/豆粕.xlsx"],
                # "棕榈油": ["基差/棕榈油.xlsx"],
                # "焦煤": ["基差/焦煤.xlsx"],
            },
            "factors": [BasisRate()],
        }
    },
    "technical": {
        "technical": {
            "factors": [
                LastDayClosePrice(),
                OISum(),
                AggAtr(),
                ROC([0]),
                ROCNext(),
                RGap(),
                RangeLnHL(),
                CLV(),
                DLogVol(),
                ATR([14]),
                ATRRelative([14]),
                DEVLogVol([7]),
                EMA([7]),
                EMA([12]),
                EMA([26]),
                EMA([9]),
                EMA([21]),
                DEVEma([7]),
                Slope([7, 14]),
                Spread([7, 21]),
                DEVEmaQuantile([60, 80]),
                DEVEmaQuantile([60, 20]),
                NumEffectiveDays([60]),
                HiEvent(),
                LoEvent(),
                # MACDLine(12, 26),
                # MACDSignal(9),
                MACDHist([12, 26, 9]),
                MACDHistNorm(),
                MACDHistCrossEvent(),
                BBandsUpper([20]),
                BBandsMiddle([20]),
                BBandsLower([20]),
                BBandsWidth(),
                BBandsPercentB(),
                BBandsWidthRelative(),
                BBandsBreakHigh(),
                BBandsBreakLow(),
                ADX([14]),
                RSI([14]),
                RSICenter(),
                RSIHiEvent(),
                RSILoEvent(),
                HV([15, 10]),
                d_HV(),
                ChaikinVolatility([10]),
                OBVFlow([7]),
                Mom([10]),
                RngRel(),
                RngMean([10]),
                DRngMean(),
                UpDownNext(),
                RelativeToMainLogDiff(),
            ],
        },
    },
    "predict": {
        "predict": {
            "factors": [
                RelativeToPred(),
                DiffAbsAtr(),
            ],
        },
    },
}

# 碳酸锂 工业硅 沪镍 沪锡 沪铜 沪银 螺纹钢 氧化铝 橡胶 纯碱 烧碱 玻璃 甲醇 精对苯二甲酸 铁矿石 豆粕 棕榈油 焦煤


def expand_weekly_data_to_daily(weekly_df):
    def expand_week(row):
        match = re.search(
            r"\((\d{4}-\d{2}-\d{2})\s*--\s*(\d{4}-\d{2}-\d{2})\)", row["周度"]
        )
        if not match:
            return None
        start_date, end_date = match.groups()
        dates = pd.date_range(start=start_date, end=end_date)

        return pd.DataFrame({"日期": dates.strftime("%Y-%m-%d"), "产量": row["产量"]})

    expanded_df = pd.concat(
        weekly_df.apply(expand_week, axis=1).dropna().reset_index(drop=True).tolist(),
        ignore_index=True,
    )
    return expanded_df


def factor_compute(func: BaseProcessor, var: pd.DataFrame, args: list):
    new_factor_cal_df = pd.DataFrame(
        columns=["合约代码", "交易日期", "factor_placeholder"]
    )
    if func.is_aggregated():
        args[0] = var
        new_factor_cal_df = func.run(args)
    else:
        contract_groups = var.groupby("合约代码")
        for contract_name, contract in contract_groups:
            args[0] = contract
            dfs = [new_factor_cal_df]
            dfs = [df for df in dfs if not df.empty and not df.isna().all().all()]

            new_factor_cal_df = (
                pd.concat(
                    [new_factor_cal_df, func.run(args)], axis=0, ignore_index=True
                )
                if not new_factor_cal_df.empty
                else func.run(args)
            )

    return new_factor_cal_df


# def compute_profit_factors(var: pd.DataFrame, var_name: str):
#     factors = FUNDAMENTALS_FACTORS[FactorCategory.PROFIT]
#     if var_name not in PROFIT:
#         return

#     for factor_name, func in factors.items():
#         for ext_name, profit_spotprice_files in (
#             ("profit", f) for f in PROFIT[var_name]
#         ):
#             if factor_name == "ProfitMargin":
#                 profit_spotprice_file, spot_price_file = profit_spotprice_files
#                 profit_var = pd.read_excel(f"InputDir/{profit_spotprice_file}")
#                 spot_price_var = pd.read_excel(f"InputDir{spot_price_file}")
#                 profit_var_dfs = {}
#                 spot_price_var_dfs = {}
#                 for col in profit_var.columns[1:]:  # skip the first column
#                     sub_df = profit_var[["日期", col]].copy()
#                     profit_var_dfs[col] = sub_df

#                 for col in spot_price_var.columns[1:]:
#                     sub_df = spot_price_var[["指标名称", col]].copy()
#                     spot_price_var_dfs[col] = sub_df

#                 for (k1, v1), (k2, v2) in itertools.product(
#                     profit_var_dfs.items(), spot_price_var_dfs.items()
#                 ):
#                     if k1 in [
#                         "碳酸锂--电渗析法(元/吨)",
#                         "碳酸锂--硫酸法(元/吨)",
#                     ] or k2 in ["中国:平均价:碳酸锂(电池级,99.5%,国产)"]:
#                         continue

#                     profit_var = canonicalize_factor_cal_df(v1)
#                     spot_price_var = canonicalize_factor_cal_df(v2)
#                     var, profit_var, spot_price_var = unification_dates(
#                         var, profit_var, spot_price_var
#                     )
#                     assert len(var) == len(profit_var) == len(spot_price_var)
#                     var[factor_name] = func(profit_var, spot_price_var)

#             elif factor_name.startswith("DailyProfit"):
#                 profit_spotprice_file, _ = profit_spotprice_files
#                 profit_var = canonicalize_factor_cal_df(profit_var)
#                 var, profit_var = unification_dates(var, profit_var)
#                 assert len(var) == len(profit_var)
#                 var[factor_name] = func(var, profit_var)


def all_factor_compute(factors: list[str], var_all: pd.DataFrame, var_name: str):
    for factor_category in factors:
        for factor_subcategory_name, factor_subcategory in CONFIG[
            factor_category
        ].items():
            for factor_func in factor_subcategory["factors"]:
                factor_name = factor_func.name()
                args = [None]
                files = factor_subcategory.get("files")
                if files:
                    args.extend(
                        [pd.read_excel(f"InputDir/{file}") for file in files[var_name]]
                    )
                new_factor_cal_df = factor_compute(
                    factor_func,
                    var_all,
                    args,
                )
                identity = factor_name
                new_factor_cal_df = new_factor_cal_df.rename(
                    columns={"factor_placeholder": identity}
                )
                var_all = pd.merge(
                    var_all,
                    new_factor_cal_df,
                    on=["合约代码", "交易日期"],
                    how="outer",
                )
    return var_all


def ts_ic_rolling(dates, vals_f, vals_r, win=50):

    out = []
    if len(dates) < win:
        return pd.DataFrame(columns=["开始日期", "结束日期", "N", "IC", "p"])

    for i in range(win - 1, len(dates)):
        f_win = vals_f[i - win + 1 : i + 1]
        r_win = vals_r[i - win + 1 : i + 1]
        # Spearman
        rho, pval = spearmanr(f_win, r_win, nan_policy="omit")

        out.append(
            {
                "开始日期": dates[i - win + 1],
                "结束日期": dates[i],
                "Days": len(f_win[np.isfinite(f_win) & np.isfinite(r_win)]),
                "IC": rho,
                "P": pval,
            }
        )
    return pd.DataFrame(out)


def ts_ic_compute(factors: list[str], var_all: pd.DataFrame, var_name: str):
    for factor_category in factors:
        for factor_subcategory_name, factor_subcategory in CONFIG[
            factor_category
        ].items():
            for factor_func in factor_subcategory["factors"]:
                factor_name = factor_func.name()
                var_date_factor_rocnext = (
                    var_all[["交易日期", factor_name, "ROCNext"]]
                    .dropna()
                    .reset_index(drop=True)
                )
                ts_ic_rolling_result = ts_ic_rolling(
                    var_date_factor_rocnext["交易日期"],
                    var_date_factor_rocnext[factor_name],
                    var_date_factor_rocnext["ROCNext"],
                )
                print(ts_ic_rolling_result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="InputDir/all_trading_date_merge.csv",
    )
    parser.add_argument("--vars", type=str, nargs="+", default=None)

    parser.add_argument("--model_training", action="store_true", default=False)
    parser.add_argument("--factor_mining", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)

    parser.add_argument("--factors", type=str, nargs="+", default=None)

    args = parser.parse_args()

    INPUT_PATH = args.input_path
    IS_DEBUG = args.debug

    profit_var = pd.read_csv(INPUT_PATH, encoding="utf-8")

    factor_mining = args.factor_mining
    model_training = args.model_training

    major = utils.prepoccess_data(profit_var, only_keep_major=factor_mining)

    grouped = major.groupby("品种名称")

    all_var_factors = None

    for var_name in args.vars:
        var_indices = grouped.groups[var_name]
        var_current_data = major.loc[var_indices].copy().reset_index(drop=True)
        var_current_data = var_current_data.sort_values(["交易日期", "合约代码"])
        # var_all.to_csv(f"InputDir/{var_name}_raw_df.csv", index=False)
        # trading_date_list = var_all["交易日期"].unique().tolist()
        # # dump to a txt file
        # with open(f"InputDir/{var_name}_trading_date_list.txt", "w") as f:
        #     for date in trading_date_list:
        #         f.write(date.strftime("%Y-%m-%d") + "\n")

        if args.factors:
            var_current_data = all_factor_compute(
                args.factors, var_current_data, var_name
            )
            all_var_factors = (
                pd.concat([all_var_factors, var_current_data], axis=0)
                if all_var_factors is not None
                else var_current_data.copy()
            )

        # create a csv file to store the predict result,which have three columns
        pred_rows_buffer = []

        if factor_mining:
            ts_ic_compute(args.factors, var_current_data, var_name)

        if model_training:
            for cut_off_date in global_var.trading_date_list:
                var = utils.filter_contracts(var_current_data, cut_off_date)
                var = utils.update_predict_contract(var, cut_off_date)
                var = all_factor_compute(["predict"], var, var_name)
                var = var.sort_values(["交易日期", "合约代码"])
                var.drop(
                    columns=global_var.drop_columns,
                    inplace=True,
                )
                if IS_DEBUG:
                    var.to_csv(
                        f"{global_var.model_debug_dir}/{cut_off_date.strftime('%Y-%m-%d')}py_pre.csv",
                        index=False,
                    )
                var = var.dropna(
                    subset=var.columns.difference(["UpDownNext", "ROCNext"])
                ).loc[~np.isinf(var["ROCNext"])]

                predict_element = var[
                    (var["交易日期"] == cut_off_date) & (var["IsPredicted"] == 1)
                ]
                var = var[var["交易日期"] != cut_off_date]
                if not IS_DEBUG:
                    var.to_csv(
                        f"{global_var.model_debug_dir}/{cut_off_date.strftime('%Y-%m-%d')}py_post.csv",
                        index=False,
                    )
                # get the last 60% data of var to assign to var
                var = var.iloc[-int(len(var) * 0.6) :]

                xgb_model = model.run_walkforward(var, cut_off_date, IS_DEBUG)
                raw_preds, _ = model.predict(
                    xgb_model, predict_element, cut_off_date, IS_DEBUG
                )
                predict_res = (
                    "涨"
                    if raw_preds > global_var.UP_TH
                    else "跌" if raw_preds < global_var.DN_TH else "无"
                )
                actual_value = (
                    "涨"
                    if predict_element["ROCNext"].iloc[0] > 0
                    else ("跌" if predict_element["ROCNext"].iloc[0] < 0 else "无")
                )
                print(cut_off_date, predict_res, actual_value, raw_preds)
                row = {
                    "交易日期": cut_off_date,
                    "预测值": predict_res,
                    "实际值": actual_value,
                    "P": raw_preds[0],
                }
                pred_rows_buffer.append(row)

            predict_result_df = pd.DataFrame(
                pred_rows_buffer, columns=["交易日期", "预测值", "实际值", "P"]
            )
            if IS_DEBUG:
                predict_result_df.to_csv(f"{var_name}_predict_result.csv", index=False)

    if factor_mining:
        # 1. compute factor ic
        # 2. compute factor ts ic
        # 3. compute factor cross-sectional ic
        pass


if __name__ == "__main__":
    main()
