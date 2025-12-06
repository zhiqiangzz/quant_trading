import pandas as pd

# load output.csv
df_R = pd.read_csv("R_contract_totrain.csv")
df_Py = pd.read_csv("Py_contract_totrain.csv")
# df_ground_truth_pre_drop = pd.read_csv("ground_truth_pre_drop.csv")

# Set index so we can find intersection
df_R.set_index(["交易日期", "合约代码"], inplace=True)
df_Py.set_index(["交易日期", "合约代码"], inplace=True)

# 在df_ground_truth中但不在df中的记录
diff_R = df_R.index.difference(df_Py.index)
# 在df中但不在df_ground_truth中的记录
diff_Py = df_Py.index.difference(df_R.index)

# Filter and save both
df_R_only = df_R.loc[diff_R].reset_index()
df_Py_only = df_Py.loc[diff_Py].reset_index()
# df_mined_diff = df_mined_diff[
#     pd.to_datetime(df_mined_diff["交易日期"]) < pd.to_datetime("2025/01/06")
# ]

# df_mined_diff.set_index(["交易日期", "合约代码"], inplace=True)
# df_ground_truth_pre_drop["交易日期"] = pd.to_datetime(
#     df_ground_truth_pre_drop["交易日期"]
# ).dt.normalize()
# df_ground_truth_pre_drop.set_index(["交易日期", "合约代码"], inplace=True)
# common_index = df_ground_truth_pre_drop.index.intersection(df_mined_diff.index)
# debug_tmp_df = df_ground_truth_pre_drop.loc[common_index].reset_index()
# debug_tmp_df.to_csv("debug_tmp_df.csv", index=False)


df_R_only.to_csv("R_contract_totrain_only.csv", index=False)
df_Py_only.to_csv("Py_contract_totrain_only.csv", index=False)
