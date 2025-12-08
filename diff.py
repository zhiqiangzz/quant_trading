import pandas as pd
import argparse

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--R", type=str, default="R_post.csv")
parser.add_argument("--Py", type=str, default="Py_post.csv")
args = parser.parse_args()

# load output.csv
df_R = pd.read_csv(args.R)
df_Py = pd.read_csv(args.Py)
# df_ground_truth_pre_drop = pd.read_csv("ground_truth_pre_drop.csv")

# Set index so we can find intersection
df_R.set_index(
    ["合约代码", "交易日期"],
    inplace=True,
)
df_Py.set_index(["合约代码", "交易日期"], inplace=True)

# 在df_ground_truth中但不在df中的记录
diff_R = df_R.index.difference(df_Py.index)
# 在df中但不在df_ground_truth中的记录
diff_Py = df_Py.index.difference(df_R.index)

common_R = df_R.index.intersection(df_Py.index)
common_Py = df_Py.index.intersection(df_R.index)


# Filter and save both
df_R_only = df_R.loc[diff_R].reset_index()
df_Py_only = df_Py.loc[diff_Py].reset_index()

df_common_R = df_R.loc[common_R].reset_index()
df_common_Py = df_Py.loc[common_Py].reset_index()

df_common_R.to_csv("R_common.csv", index=False)
df_common_Py.to_csv("Py_common.csv", index=False)

df_R_only.to_csv("R_only.csv", index=False)
df_Py_only.to_csv("Py_only.csv", index=False)
