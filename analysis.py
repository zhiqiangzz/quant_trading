import pandas as pd
from datetime import datetime


factor_list = ["DEMA_5d", "DEMA_10d", "EMA_5d", "EMA_10d", "LogReturn"]
# factor_list = ["DEMA_5d"]
variant_list = [
    "工业硅",
    "棕榈油",
    "橡胶",
    "沪铜",
    "沪银",
    "沪锡",
    "沪镍",
    "焦煤",
    "玻璃",
    "甲醇",
    "精对苯二甲酸",
    "纯碱",
    "螺纹钢",
    "豆粕",
    "铁矿石",
    "氧化铝",
    "碳酸锂",
    "烧碱",
]


def merge_date_regions(regions):
    """
    Merge continuous or overlapping date regions that have the same relation grade and confidence.

    Args:
        regions: List of dictionaries with 'start_date', 'end_date', 'relation_grade', 'relation_confidence', 'mean'

    Returns:
        List of merged regions
    """
    if not regions:
        return []

    # Sort regions by start date
    sorted_regions = sorted(regions, key=lambda x: x["start_date"])
    merged = [sorted_regions[0]]

    for current in sorted_regions[1:]:
        last_merged = merged[-1]

        # Check if regions can be merged (same grade/confidence and overlapping or continuous)
        can_merge = (
            last_merged["relation_grade"] == current["relation_grade"]
            and last_merged["relation_confidence"] == current["relation_confidence"]
            and (
                current["start_date"] <= last_merged["end_date"] + pd.Timedelta(days=1)
            )  # Allow 1-day gap
        )

        if can_merge:
            # Merge: extend the end date and update mean
            merged[-1]["end_date"] = max(last_merged["end_date"], current["end_date"])
            merged[-1]["mean"] = (
                last_merged["mean"] + current["mean"]
            ) / 2  # Average the means
        else:
            # Cannot merge: add as new region
            merged.append(current)

    return merged


def main():
    for factor in factor_list:
        with open(f"{factor}.md", "w") as f:
            input_dir = factor
            input_cs_daily_path = f"{input_dir}/{factor}_CSIC_按天.csv"
            input_cs_summary_path = f"{input_dir}/{factor}_CSIC_汇总.csv"
            input_cs_roll_60_path = f"{input_dir}/{factor}_CSIC_滚动60.csv"

            df_cs_ic_daily = pd.read_csv(input_cs_daily_path, encoding="utf-8")
            df_cs_ic_summary = pd.read_csv(input_cs_summary_path, encoding="utf-8")
            df_cs_ic_roll_60 = pd.read_csv(input_cs_roll_60_path, encoding="utf-8")
            overall_mean = df_cs_ic_summary["MEAN"].iloc[0]
            std = df_cs_ic_summary["STD"].iloc[0]
            icir = df_cs_ic_summary["ICIR"].iloc[0]
            t = df_cs_ic_summary["T"].iloc[0]
            p = df_cs_ic_summary["P"].iloc[0]
            f.write(
                f"""## {factor} 单因子分析报告
分析时间: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n
"""
            )
            f.write(f"### 整体表现评估:\n")
            # print as the markdown table format
            f.write("| 均值 | 标准差 | ICIR | T | P |\n")
            f.write("| -------- | -------- | -------- | -------- | -------- |\n")
            f.write(f"| {overall_mean} | {std} | {icir} | {t} | {p} |\n")
            f.write("\n")
            f.write("总体上, ")
            if overall_mean >= 0.01:
                f.write("均值 >= 0.01 较强相关性\n")
            else:
                f.write("均值 < 0.01 弱相关性，接近于0\n")
            if icir >= 0.3:
                f.write("ICIR >= 0.3 稳定度较高\n")
            else:
                f.write("ICIR < 0.3 稳定度较低\n")
            if t > 2:
                f.write("T > 2 显著性较高\n")
            else:
                f.write("T <= 2 显著性较低\n")
            f.write("\n")

            f.write(f"""### 时间序列表现分析:\n""")
            df_cs_ic_daily = df_cs_ic_daily.sort_values(by="CS_IC", ascending=False)
            f.write(f"""TOP10 相关性最强且显著性较高:\n\n""")
            count = 0
            # print as the markdown table format
            f.write("| date | CS_IC | P |\n")
            f.write("| -------- | -------- | -------- |\n")
            for _, row in df_cs_ic_daily.iterrows():
                if row["CS_P"] <= 0.1 and row["CS_IC"] >= 0.02 and count < 10:
                    count += 1
                    f.write(f"| {row['交易日期']} | {row['CS_IC']} | {row['CS_P']} |\n")
            f.write("\n")

            f.write(f"""### 时间序列窗口表现分析:\n""")
            for _, row in df_cs_ic_roll_60.iterrows():
                date_begin = row["开始日期"]
                date_end = row["结束日期"]
                mean = row["IC"]
                iclr = row["ICIR"]
                t = row["T"]
                p = row["P"]
                if abs(mean) >= 0.01 and abs(iclr) >= 0.3 and abs(t) > 2:
                    f.write(
                        f"{date_begin} - {date_end} |均值{mean}| >= 0.01 较强相关性, |ICIR {iclr}| >= 0.3 稳定度较高, |T {t}| > 2 显著性较高\n"
                    )

            tsic_summary_file_name = f"{input_dir}/{factor}_TSIC_整体_各品种.csv"
            df_ts_ic_summary = pd.read_csv(tsic_summary_file_name, encoding="utf-8\n")
            # overall = df_ts_ic_summary[df_ts_ic_summary["品种名称"] == "整体"]
            # overall_mean = overall["TS_IC"].iloc[0]
            # p = overall["TS_P"].iloc[0]
            f.write("\n")
            f.write(f"### 品种表现分析:\n")
            # print as the markdown table format
            f.write("| 品种名称 | 均值 | P |\n")
            f.write("| -------- | -------- | -------- |\n")
            for _, row in df_ts_ic_summary.iterrows():
                f.write(f"| {row['品种名称']} | {row['TS_IC']} | {row['TS_P']} |\n")
            f.write("\n")
            f.write("总体上, ")
            overall_abs_mean = abs(overall_mean)
            if overall_mean < 0:
                f.write(f"均值 {overall_mean} < 0 整体相关性为负\n")
            else:
                f.write(f"均值 {overall_mean} >= 0 整体相关性为正\n")

            if overall_abs_mean >= 0.02 and overall_abs_mean <= 0.3:
                f.write(f"0.02 < |{overall_mean}| <= 0.3 总体相关性较强\n")
            elif overall_abs_mean > 0.3:
                f.write(f"|{overall_mean}| > 0.3 总体相关性很强\n")
            elif overall_abs_mean <= 0.02:
                f.write(f"|{overall_mean}| < 0.2 总体相关性较弱\n")
            if p <= 0.1:
                f.write(f"P <= 0.1 总体 显著性较高\n\n")
            else:
                f.write(f"P > 0.1 总体 显著性较低\n\n")

            f.write("### 滚动窗口分析:\n")
            for variant_name in variant_list:
                tsic_roll_file_name = f"{input_dir}/{variant_name}_TSIC_滚动50.csv"
                df_tsic_roll = pd.read_csv(tsic_roll_file_name, encoding="utf-8\n")
                variant = df_ts_ic_summary[df_ts_ic_summary["品种名称"] == variant_name]
                if variant.empty:
                    continue
                mean = variant["TS_IC"].iloc[0]
                abs_mean = abs(mean)
                p = variant["TS_P"].iloc[0]
                relation_grade = 0
                relation_confidence = 0
                if abs_mean > 0.3:
                    relation_grade = 2
                    f.write(
                        f"- **{variant_name} 总体呈现很强{"负" if mean < 0 else "正"}相关性**",
                    )
                elif abs_mean >= 0.02 and abs_mean <= 0.3:
                    relation_grade = 1
                    f.write(
                        f"- {variant_name} 总体呈现较强{"负" if mean < 0 else "正"}相关性",
                    )
                elif abs_mean <= 0.02:
                    relation_grade = 0
                    f.write(
                        f"- {variant_name} 总体呈现弱{"负" if mean < 0 else "正"}相关性",
                    )

                if p <= 0.1:
                    relation_confidence = 1
                    f.write(f" {"且" if relation_grade > 0 else "但"}**显著性较高**\n")
                else:
                    relation_confidence = 0
                    f.write(f" {"且" if relation_grade == 0 else "但"}显著性较低\n")

                # Process rolling data and merge continuous regions
                regions = []
                for _, row in df_tsic_roll.iterrows():
                    date_begin = row["开始日期"]
                    date_end = row["结束日期"]
                    mean = row["IC"]
                    p = row["P"]
                    abs_mean = abs(mean)
                    relation_grade = 0
                    if abs_mean > 0.3:
                        relation_grade = 2
                    elif abs_mean >= 0.02 and abs_mean <= 0.3:
                        relation_grade = 1

                    if p <= 0.1:
                        relation_confidence = 1
                    else:
                        relation_confidence = 0

                    # Only consider significant and meaningful correlations
                    if relation_grade > 0 and relation_confidence == 1:
                        regions.append(
                            {
                                "start_date": pd.to_datetime(date_begin),
                                "end_date": pd.to_datetime(date_end),
                                "relation_grade": relation_grade,
                                "relation_confidence": relation_confidence,
                                "mean": mean,
                            }
                        )

                # Merge continuous or overlapping regions
                if regions:
                    merged_regions = merge_date_regions(regions)

                    # Categorize by relation grade
                    strong_regions = [
                        r for r in merged_regions if r["relation_grade"] == 2
                    ]
                    moderate_regions = [
                        r for r in merged_regions if r["relation_grade"] == 1
                    ]

                    # f.write strong correlation regions (grade == 2)
                    if strong_regions:
                        f.write(
                            f"  **很强相关性且显著性明显时间段** (共{len(strong_regions)}个):"
                        )
                        line = "、".join(
                            f"{region['start_date'].strftime('%Y-%m-%d')} - {region['end_date'].strftime('%Y-%m-%d')}"
                            for region in strong_regions
                        )
                        f.write(line)
                        f.write("\n")

                    # f.write moderate correlation regions (grade == 1)
                    if moderate_regions:
                        f.write(
                            f"  *较强相关性且显著性明显时间段* (共{len(moderate_regions)}个):"
                        )
                        line = "、".join(
                            f"{region['start_date'].strftime('%Y-%m-%d')} - {region['end_date'].strftime('%Y-%m-%d')}"
                            for region in moderate_regions
                        )
                        f.write(line)
                        f.write("\n")

                    # Summary
                    if not strong_regions and not moderate_regions:
                        f.write("  无显著相关性时间段\n")


if __name__ == "__main__":
    main()
