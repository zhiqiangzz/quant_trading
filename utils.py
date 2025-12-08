import numpy as np
import pandas as pd
import os
import re
from scipy.stats import ttest_1samp
from scipy.stats import spearmanr
from scipy.stats import spearmanr, t as t_metric


# ========= 工具函数 =========
def to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def icir(ic_series):
    x = pd.Series(ic_series).dropna()
    T = len(x)
    if T == 0:
        return np.nan
    return x.mean() / x.std(ddof=1)


def t_value(ic_series):
    x = pd.Series(ic_series).dropna()
    T = len(x)
    if T == 0:
        return np.nan
    if T == 1:
        return 0
    std_value = x.std(ddof=1)
    assert std_value >= 0
    return x.mean() / (std_value / np.sqrt(T))


def p_value(ic_series):
    x = pd.Series(ic_series).dropna()
    T = len(x)
    if T == 0:
        return np.nan
    return 2 * (1 - t_metric.cdf(abs(t_value(x)), df=T - 1))


def cs_ic_summary(ic_series):
    """
    对 IC 的日度序列做总体汇总：T, mean, sd, ICIR, t（独立近似）
    """
    ic_series = pd.Series(ic_series).dropna()
    T = len(ic_series)
    assert T > 0
    mean_value = ic_series.mean()
    std_value = ic_series.std(ddof=1)
    icir_value = icir(ic_series)
    t_stat, p_val = ttest_1samp(ic_series, 0)
    return {
        "MEAN": mean_value,
        "STD": std_value,
        "ICIR": icir_value,
        "T": t_stat,
        "P": p_val,
    }


def ts_ic_rolling(df_prod, factor_col, ret_col, win=50):
    """
    对“单一品种”的时间序列做 50 日滚动 TS-IC（Spearman）
    返回：窗口表（开始/结束日期, N, IC, p）
    """
    sub = df_prod[["交易日期", factor_col, ret_col]].dropna().copy()
    out = []
    vals_f = sub[factor_col].to_numpy()
    vals_r = sub[ret_col].to_numpy()
    dates = sub["交易日期"].to_numpy()

    if len(sub) < win:
        return pd.DataFrame(columns=["开始日期", "结束日期", "N", "IC", "p"])

    for i in range(win - 1, len(sub)):
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


def cs_ic_rolling(df_daily, win=60):
    out = []
    dates = df_daily["交易日期"].to_numpy()
    for i in range(win - 1, len(df_daily)):
        cs_ic_windows = df_daily["CS_IC"][i - win + 1 : i + 1]
        ic_mean = np.mean(cs_ic_windows)
        t_stat, p_val = ttest_1samp(cs_ic_windows, 0)
        icir_val = icir(cs_ic_windows)

        out.append(
            {
                "开始日期": dates[i - win + 1],
                "结束日期": dates[i],
                "Days": win,
                "IC": ic_mean,
                "ICIR": icir_val,
                "T": t_stat,
                "P": p_val,
            }
        )
    return pd.DataFrame(out)


def canonicalize_datetime_column(df):
    df = df[pd.to_datetime(df["交易日期"], errors="coerce").notna()].reset_index(
        drop=True
    )
    df["交易日期"] = pd.to_datetime(df["交易日期"]).dt.normalize()
    # sort by trading date
    df = df.sort_values("交易日期")

    return df


def filter_contracts(df, selected_date):
    # extract year and month from selected_date
    year = selected_date.year % 100
    month = selected_date.month
    day = selected_date.day
    # set the begin and end of the kept range
    begin_date = (year - 1, month)
    month = month + 9 if day < 11 else month + 10
    year = year if month <= 12 else year + 1
    month = month % 13
    end_date = (year, month)

    def is_contract_in_range(contract_str):
        if pd.isna(contract_str):
            return False

        contract_str = str(contract_str).lower()

        match = re.search(r"(\d{2})(\d{2})", contract_str)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            return begin_date <= (year, month) < end_date

        return False

    df = df[df["交易日期"] <= selected_date]
    mask = df["合约代码"].apply(is_contract_in_range)
    filtered_df = df[mask].copy()

    return filtered_df

    # # 工具：取某年某月的第 n 个交易日（不足 n 天则 NA）
    # nth_trading_day <- function(y, m, n = 10L) {
    #   start <- as.Date(sprintf("%04d-%02d-01", y, m))
    #   end   <- ceiling_date(start, "month") - days(1)
    #   days_in_month <- td_all$交易日[td_all$交易日 >= start & td_all$交易日 <= end]
    #   if (length(days_in_month) >= n) days_in_month[n] else as.Date(NA)
    # }

    # # ========== 1) 基于 df_all 解析合约、交割日、DaysToExpiry ==========
    # ## 先确保类型正确 ---------------------------------------------
    # ## 1) 解析合约代码：前缀 / 年(YY) / 月(MM) ------------------------
    # parsed <- strcapture(
    #   pattern = "^([A-Za-z]+)(\\d{2})(\\d{2})",
    #   x       = df_all$合约代码,
    #   proto   = list(前缀 = character(), YY = integer(), MM = integer())
    # )

    # ## 2) 计算交割日（该合约 YYYY-MM 的第 10 个交易日） ---------------
    # YYYY <- ifelse(!is.na(parsed$YY), 2000L + parsed$YY, NA_integer_)
    # MM    <- ifelse(!is.na(parsed$MM) & parsed$MM >= 1 & parsed$MM <= 12, parsed$MM, NA_integer_)

    # expiry_vec <- mapply(nth_trading_day, YYYY, MM, MoreArgs = list(n = 10L))
    # expiry_vec <- as.Date(expiry_vec, origin = "1970-01-01")

    # ## 3) DaysToExpiry：若当日 >= 交割日 → 0；否则计数 (当日, 交割日] 的交易日个数
    # days_to_expiry <- mapply(function(d1, d2) {
    #   if (is.na(d2)) return(NA_integer_)
    #   if (!is.na(d1) && d1 >= d2) return(0L)
    #   sum(td_all$交易日 > d1 & td_all$交易日 <= d2)
    # }, df_all$交易日期, expiry_vec)

    # ## 4) 合并回 df_all ----------------------------------------------
    # df_all$前缀      <- parsed$前缀
    # df_all$YY        <- parsed$YY
    # df_all$MM        <- MM
    # df_all$YYYY      <- YYYY
    # df_all$交割日    <- as.Date(expiry_vec)
    # df_all$DaysToExpiry <- as.integer(days_to_expiry)


expiry_date_cache = {}


def compute_days_to_expiry(df):
    expiry_dfs = [
        pd.read_excel(f"InputDir/交易日期/{year}年交易日列表.xlsx").rename(
            columns={"日期": "交易日期", "交易日": "交易日期"}
        )
        for year in range(2023, 2027)
    ]
    expiry_dfs = pd.concat(expiry_dfs)
    expiry_dfs = canonicalize_datetime_column(expiry_dfs)

    def get_expiry_date(contract_code):
        match = re.search(r"(\d{2})(\d{2})", contract_code)
        if match:
            year = 2000 + int(match.group(1))
            month = int(match.group(2))
            if (year, month) in expiry_date_cache:
                return expiry_date_cache[(year, month)]

            target_month_trading_dates_list = (
                expiry_dfs[
                    (expiry_dfs["交易日期"].dt.year == year)
                    & (expiry_dfs["交易日期"].dt.month == month)
                ]["交易日期"]
                .sort_values()
                .tolist()
            )
            expiry_date_cache[(year, month)] = (
                target_month_trading_dates_list[9]
                if len(target_month_trading_dates_list) >= 10
                else (
                    target_month_trading_dates_list[-1]
                    if target_month_trading_dates_list
                    else None
                )
            )
            return expiry_date_cache[(year, month)]
        assert False

    df["交割日"] = df["合约代码"].apply(get_expiry_date)

    df["DaysToExpiry"] = df.apply(
        lambda x: (
            len(
                expiry_dfs[
                    (expiry_dfs["交易日期"] <= x["交割日"])
                    & (expiry_dfs["交易日期"] > x["交易日期"])
                ]
            )
        ),
        axis=1,
    )
    # drop 交割日
    df = df.drop(columns=["交割日"])
    return df


def prepoccess_data(df, EXCLUDE_SYMS={"多晶硅"}, only_keep_major=False):
    df = to_numeric(df, ["开盘价", "收盘价", "成交量", "持仓量", "结算价", "前结算价"])

    df = df[~df["品种名称"].isin(EXCLUDE_SYMS)].copy()
    df = df.sort_values(["合约代码", "交易日期"])
    df = canonicalize_datetime_column(df)
    df = compute_days_to_expiry(df)
    df["前收盘价"] = df.groupby("合约代码")["收盘价"].shift(1)

    # delete row with any item is NA
    df = df.dropna(
        subset=["开盘价", "收盘价", "成交量", "持仓量", "结算价", "前结算价"]
    )
    # delete row with 开盘价 is 0
    df = df[df["开盘价"] != 0]

    idx = df.groupby(["交易日期", "品种名称"])["成交量"].idxmax()
    df["IsMain"] = df.index.isin(idx).astype(int)
    if only_keep_major:
        df = df[df["IsMain"] == 1]
        df["合约代码"] = "major_contract_placeholder"

    # # predicted is the first trading date after cut_off_date
    # # add extra column IsPredicted to indicate if the contract of that row is same as the predicted contract
    # if cut_off_date is not None:
    #     predicted_date = df[df["交易日期"] >= cut_off_date]["交易日期"].min()
    #     # set the predicted contract list of the predicted date and IsMain is 1
    #     predict_contract_list = df[
    #         (df["交易日期"] == predicted_date) & (df["IsMain"] == 1)
    #     ]["合约代码"].tolist()
    #     df["IsPredicted"] = df["合约代码"].isin(predict_contract_list)

    return df


def update_predict_contract(df, cut_off_date):
    predicted_date = df[df["交易日期"] == cut_off_date]
    # set the predicted contract list of the predicted date and IsMain is 1
    predict_contract_list = predicted_date[predicted_date["IsMain"] == 1][
        "合约代码"
    ].tolist()
    assert len(predict_contract_list) == 1
    df["IsPredicted"] = df["合约代码"].isin(predict_contract_list)
    return df


def correlation_calculation(
    major,
    factor,
    var_extension_name=None,
    is_cs=True,
    TS_ROLL_WIN=50,
    CS_ROLL_WIN=60,
    CS_MIN_N=12,
    dump=True,
):
    # ========= 逐品种：50 日滚动 TS-IC 表 + 整体 TS-IC 汇总 =========
    os.makedirs(f"OutputDir/{factor}", exist_ok=True)
    per_var_roll = {}
    per_var_overall = []

    vars = major.groupby("品种名称")

    for var, g in vars:
        g = g.sort_values("交易日期")
        roll_tbl = ts_ic_rolling(g, factor, f"ret_o2c_tplus1", win=TS_ROLL_WIN)
        per_var_roll[var] = roll_tbl
        var_name = f"{var}{var_extension_name if var_extension_name else ''}"

        # 保存每个品种的窗口表
        out_path = os.path.join(
            f"OutputDir/{factor}",
            f"{var_name}_TSIC_滚动{TS_ROLL_WIN}.csv",
        )
        roll_tbl.to_csv(
            out_path, index=False, encoding="utf-8-sig", float_format="%.3f"
        )

        # 整体 TS-IC（该品种全时段）
        sub = g[[factor, "ret_o2c_tplus1"]].dropna()
        if len(sub) >= 10:  # 至少 10 个点再算
            rho, pval = spearmanr(sub[factor], sub["ret_o2c_tplus1"], nan_policy="omit")
        else:
            rho, pval = (np.nan, np.nan)
        per_var_overall.append(
            {
                "品种名称": var_name,
                "TS_IC": rho,
                "TS_P": pval,
                "Days": len(sub),
            }
        )

    if len(vars) > 1:
        overall_ic_mean = np.mean([row["TS_IC"] for row in per_var_overall])
        overall_p = p_value([row["TS_IC"] for row in per_var_overall])
        per_var_overall.insert(
            0,
            {
                "品种名称": "整体",
                "TS_IC": overall_ic_mean,
                "TS_P": overall_p,
                "Days": 0,
            },
        )

    # 汇总“同品种”整体指标（每个品种一个数）
    ts_ic_summary = pd.DataFrame(per_var_overall)

    # append to existing csv if exists
    output_path = f"OutputDir/{factor}/TSIC_整体_各品种.csv"
    if os.path.exists(output_path):
        ts_ic_summary_ori = pd.read_csv(output_path)
        ts_ic_summary = pd.concat([ts_ic_summary_ori, ts_ic_summary])

    ts_ic_summary.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.3f",
    )

    if is_cs:
        # ========= 跨品种：按天做 CS-IC（t 日因子 vs t+1 日收益） =========
        cs_rows = []
        for dt, g in major.groupby("交易日期"):
            # CHANGE: 行级对齐 —— 同一 DataFrame 内对两列按行 dropna
            sub = g[[factor, "ret_o2c_tplus1"]].dropna()
            n = len(sub)
            if n >= CS_MIN_N:  # CHANGE: 截面样本数下限
                rho, pval = spearmanr(
                    sub[factor], sub["ret_o2c_tplus1"], nan_policy="omit"
                )
            else:
                rho, pval = (np.nan, np.nan)
            cs_rows.append({"交易日期": dt, "品种数量": n, "CS_IC": rho, "CS_P": pval})

        cs_ic_daily = pd.DataFrame(cs_rows).sort_values("交易日期")
        cs_ic_daily.to_csv(
            f"OutputDir/{factor}/{factor}_CSIC_按天.csv",
            index=False,
            encoding="utf-8-sig",
            float_format="%.3f",
        )

        # cs rolling
        cs_roll_tbl = cs_ic_rolling(cs_ic_daily, win=CS_ROLL_WIN)
        cs_roll_tbl.to_csv(
            f"OutputDir/{factor}/{factor}_CSIC_滚动{CS_ROLL_WIN}.csv",
            index=False,
            encoding="utf-8-sig",
            float_format="%.3f",
        )

        # CS-IC 总体汇总
        cs_summary = cs_ic_summary(cs_ic_daily["CS_IC"])
        cs_summary_df = pd.DataFrame([cs_summary])
        cs_summary_df.to_csv(
            f"OutputDir/{factor}/{factor}_CSIC_汇总.csv",
            index=False,
            encoding="utf-8-sig",
            float_format="%.3f",
        )

    if per_var_roll:
        example_df = per_var_roll.get(next(iter(per_var_roll)), pd.DataFrame()).head()

    if dump:
        if is_cs:
            # ========= 打印示例 =========
            print("== 跨品种 CS-IC 汇总 ==")
            print(cs_summary_df.to_string(index=False))

        print("\n== 同品种 TS-IC 整体（每个品种一行） ==")
        print(
            ts_ic_summary.head(10).to_string(index=False)
        )  # 只示例前 10 行；完整见导出 CSV

        print("\n== 示例：某个品种 50 日滚动 TS-IC 表（前 5 行） ==")
        print(example_df.to_string(index=False))
