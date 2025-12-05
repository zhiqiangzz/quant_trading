import pandas as pd

# load output.csv
df_ground_truth = pd.read_csv("ground_truth_after_drop.csv")
df_ground_truth_pre_drop = pd.read_csv("ground_truth_pre_drop.csv")

# sort by 交易日期 and 合约代码
df_ground_truth = df_ground_truth.sort_values(["交易日期", "合约代码"])

# round to 3 decimal places
df_ground_truth = df_ground_truth.round(3)

df = pd.read_csv("OutputDir/碳酸锂_after_drop_df.csv")

# Set index so we can find intersection
df_ground_truth.set_index(["交易日期", "合约代码"], inplace=True)
df.set_index(["交易日期", "合约代码"], inplace=True)

# 在df_ground_truth中但不在df中的记录
diff_ground_truth = df_ground_truth.index.difference(df.index)
# 在df中但不在df_ground_truth中的记录
diff_mined = df.index.difference(df_ground_truth.index)

# Filter and save both
df_ground_truth_diff = df_ground_truth.loc[diff_ground_truth].reset_index()
df_mined_diff = df.loc[diff_mined].reset_index()
# df_mined_diff = df_mined_diff[
#     pd.to_datetime(df_mined_diff["交易日期"]) < pd.to_datetime("2025/01/06")
# ]

df_mined_diff.set_index(["交易日期", "合约代码"], inplace=True)
df_ground_truth_pre_drop["交易日期"] = pd.to_datetime(
    df_ground_truth_pre_drop["交易日期"]
).dt.normalize()
df_ground_truth_pre_drop.set_index(["交易日期", "合约代码"], inplace=True)
common_index = df_ground_truth_pre_drop.index.intersection(df_mined_diff.index)
debug_tmp_df = df_ground_truth_pre_drop.loc[common_index].reset_index()
debug_tmp_df.to_csv("debug_tmp_df.csv", index=False)


df_ground_truth_diff.to_csv("output_ground_truth_only.csv", index=False)
df_mined_diff.to_csv("output_mined_only.csv", index=False)
