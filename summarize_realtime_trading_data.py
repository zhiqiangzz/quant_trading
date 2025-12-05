import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from itertools import product
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Songti SC"
plt.rcParams["axes.unicode_minus"] = False

PLOT_WIDTH = 14
PLOT_HEIGHT = 6
TITLE_FONT_SIZE = 14
LABEL_FONT_SIZE = 10
TEXT_FONT_SIZE = 8
LARGER_TEXT_FONT_SIZE = 14


def draw_time_series(g, title, output_dir):
    plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))

    start = g["交易日期"].min().strftime("%Y-%m-%d")
    end = g["交易日期"].max().strftime("%Y-%m-%d")

    # 柱的颜色：盈亏 > 0 红色，否则绿色
    colors = ["lightcoral" if x > 0 else "lightgreen" for x in g["盈亏"]]
    profit_patch = mpatches.Patch(color="lightcoral", label="单日盈利")
    loss_patch = mpatches.Patch(color="lightgreen", label="单日亏损")
    cum_line = Line2D([0], [0], color="lightblue", label="累计盈亏")

    plt.legend(
        handles=[profit_patch, loss_patch, cum_line],
        loc="upper left",  # 左上角
        fontsize=12,
        frameon=False,  # 去掉图例边框（可选）
    )

    # 画柱状图
    plt.bar(g["交易日期"].dt.strftime("%Y-%m-%d"), g["盈亏"], color=colors)
    plt.plot(g["交易日期"].dt.strftime("%Y-%m-%d"), g["累计盈亏"], color="lightblue")

    # 标注每根柱子数值和折线值
    for x, y, z in zip(g["交易日期"].dt.strftime("%Y-%m-%d"), g["盈亏"], g["累计盈亏"]):
        if y != 0:
            plt.text(
                x,
                y / 2,
                f"{int(y)}",
                ha="center",
                va="bottom" if y > 0 else "top",
                fontsize=TEXT_FONT_SIZE,
            )

        plt.text(
            x,
            z,
            f"{int(z)}",
            ha="center",
            va="top" if z > 0 else "bottom",
            fontsize=TEXT_FONT_SIZE,
            color="blue",
        )

    # 标题
    plt.title(f"{title} ({start} 至 {end})", fontsize=TITLE_FONT_SIZE)

    # 标签
    plt.xlabel("日期", fontsize=LABEL_FONT_SIZE)
    plt.ylabel("盈亏(元)", fontsize=LABEL_FONT_SIZE)

    # 旋转 X 轴日期
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    # 输出目录
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"{title}_盈亏.pdf")
    plt.savefig(pdf_path)
    plt.close()


def draw_variety(g, title, output_dir):
    plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    colors = ["lightcoral" if x > 0 else "lightgreen" for x in g["盈亏"]]
    plt.bar(g["品种"], g["盈亏"], color=colors)
    profit_patch = mpatches.Patch(color="lightcoral", label="单日盈利")
    loss_patch = mpatches.Patch(color="lightgreen", label="单日亏损")
    freq_item = Line2D([], [], color="blue", linestyle="--", label="交易频率")

    plt.legend(
        handles=[profit_patch, loss_patch, freq_item],
        loc="upper left",
        fontsize=12,
        frameon=False,
    )

    # 标注每根柱子数值和折线值
    for x, y, z in zip(g["品种"], g["盈亏"], g["交易频率"]):
        if y != 0:
            plt.text(
                x,
                y,
                f"{int(y)}",
                ha="center",
                va="bottom" if y > 0 else "top",
                fontsize=TEXT_FONT_SIZE,
            )
        if z != 0:
            plt.text(
                x,
                y / 2,
                f"{z*100:.0f}%",
                ha="center",
                va="top",
                fontsize=TEXT_FONT_SIZE,
                color="blue",
            )

    plt.title(f"{title}", fontsize=TITLE_FONT_SIZE)

    plt.xlabel("品种", fontsize=LABEL_FONT_SIZE)
    plt.ylabel("盈亏(元)", fontsize=LABEL_FONT_SIZE)

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"{title}.pdf")
    plt.savefig(pdf_path)
    plt.close()


def draw_pie(g, title, output_dir):
    values = g["盈亏"].abs()
    labels = g["品种"]
    colors = ["red" if v > 0 else "green" for v in g["盈亏"]]

    fig, ax = plt.subplots(figsize=(12, 12))

    # don't show the percentage if it's less than 1%
    def autopct_hide_small(pct):
        return f"{pct:.1f}%" if pct >= 1 else ""

    patches, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        autopct=autopct_hide_small,
        textprops={"fontsize": TEXT_FONT_SIZE, "color": "black"},
        pctdistance=0.7,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )

    for autotext in autotexts:
        autotext.set_fontsize(LARGER_TEXT_FONT_SIZE)
        autotext.set_weight("bold")
        autotext.set_color("white")

    plt.title("盈亏占比饼图（按绝对值）", fontsize=18)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"{title}.pdf")
    plt.savefig(pdf_path)
    plt.close()


def extract_trade_date(df_info):
    """从基本资料 sheet 提取交易日期（支持合并单元格、异常结构）"""
    # 找到包含“交易日期”的行
    row, col = (df_info == "交易日期").stack().idxmax()
    return df_info.iloc[row, col + 2]


def process_file(path):
    """处理单个 XLS 文件，返回标准化 df_trade"""
    xlsx = pd.read_excel(path, engine="xlrd", sheet_name=None, header=None)

    df_raw = xlsx["品种汇总"]
    trade_date = extract_trade_date(df_raw)

    title_row = df_raw[df_raw.eq("品种").any(axis=1)].index[0]

    # 切片数据
    df_trade = df_raw.iloc[title_row:]
    df_trade.columns = df_trade.iloc[0]
    df_trade = df_trade.iloc[1:].reset_index(drop=True)

    df_trade = df_trade.dropna(axis=1, how="all")

    df_trade = df_trade[df_trade["品种"] != "合计"]

    df_trade["交易日期"] = trade_date

    return df_trade


mapping = {
    "SI": "工业硅",
    "P": "棕榈油",
    "RU": "橡胶",
    "CU": "沪铜",
    "AG": "沪银",
    "SN": "沪锡",
    "NI": "沪镍",
    "JM": "焦煤",
    "FG": "玻璃",
    "MA": "甲醇",
    "TA": "精对苯二甲酸",
    "SA": "纯碱",
    "RB": "螺纹钢",
    "M": "豆粕",
    "AO": "氧化铝",
    "SH": "烧碱",
    "LC": "碳酸锂",
    "I": "铁矿石",
}

folder = "量化每日报表"
files = glob.glob(os.path.join(folder, "*.xls"))

all_trades = []

for f in files:
    try:
        df_trade = process_file(f)
        all_trades.append(df_trade)
    except Exception as e:
        print(f"读取失败：{f}, 原因：{e}")

df_trade_all = pd.concat(all_trades, ignore_index=True)
all_varieties = df_trade_all["品种"].unique()
all_dates = df_trade_all["交易日期"].unique()


full_grid = pd.DataFrame(
    list(product(all_varieties, all_dates)), columns=["品种", "交易日期"]
)

df_complete = pd.merge(full_grid, df_trade_all, on=["品种", "交易日期"], how="left")
df_trade_all = df_complete.fillna(0)

# group by 品种 and draw a histogram of the 平仓盈亏 ,the x axis is the 交易日期, and the y axis is the 平仓盈亏, and the title is the 平仓盈亏
df_trade_all.groupby("品种")["平仓盈亏"]
output_dir_per_variety = "OutputDir/实盘盈亏_逐品种"
output_dir_per_day = "OutputDir/实盘盈亏_逐日"
output_dir_per_day_cumulative = "OutputDir/实盘盈亏_逐日累计"
output_dir_per_variety_cumulative = "OutputDir/实盘盈亏_逐品种累计"

# print(df_trade_all)
df_trade_all["交易日期"] = pd.to_datetime(df_trade_all["交易日期"])
df_trade_all["品种"] = df_trade_all["品种"].map(mapping)
df_trade_all.sort_values("交易日期", inplace=True)
print(
    "胜率：",
    len(df_trade_all[df_trade_all["平仓盈亏"] > 0])
    / len(df_trade_all[df_trade_all["平仓盈亏"] != 0]),
)


# for variety, g in df_trade_all.groupby("品种"):
#     g = g.sort_values("交易日期")
#     g["累计盈亏"] = g["平仓盈亏"].cumsum()
#     g.rename(columns={"平仓盈亏": "盈亏"}, inplace=True)
#     draw_time_series(g, variety + "日度盈亏", output_dir_per_variety)

# daily_profit = (
#     df_trade_all.groupby("交易日期")["平仓盈亏"].sum().rename("盈亏").reset_index()
# )
# daily_profit.sort_values("交易日期", inplace=True)
# daily_profit["累计盈亏"] = daily_profit["盈亏"].cumsum()

# draw_time_series(daily_profit, "逐日累积盈亏", output_dir_per_day_cumulative)

# for date, g in df_trade_all.groupby("交易日期"):
#     g.rename(columns={"平仓盈亏": "盈亏"}, inplace=True)
#     g.sort_values("盈亏", inplace=True)
#     draw_variety(g, f"{date.strftime('%Y-%m-%d')}_各品种日度盈亏", output_dir_per_day)

var_wise_profit = (
    df_trade_all.groupby("品种")
    .agg(
        盈亏=("平仓盈亏", lambda x: x.sum()),
        交易频率=("手数", lambda x: (x != 0).mean()),
    )
    .reset_index()
)
var_wise_profit.sort_values("盈亏", inplace=True)
print(var_wise_profit)

draw_variety(var_wise_profit, "逐品种累积盈亏", output_dir_per_variety_cumulative)
# draw_pie(var_wise_profit, "逐品种累积盈亏饼图", output_dir_per_variety_cumulative)
