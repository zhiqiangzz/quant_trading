import pandas as pd

# load output.csv
df_ground_truth = pd.read_csv("output.csv")

# sort by 交易日期 and 合约代码
df_ground_truth = df_ground_truth.sort_values(["合约代码", "交易日期"])

# round to 3 decimal places
df_ground_truth = df_ground_truth.round(3)

df = pd.read_csv("OutputDir/碳酸锂_merged_df.csv")

# Set index so we can find intersection
df_ground_truth.set_index(["合约代码", "交易日期"], inplace=True)
df.set_index(["合约代码", "交易日期"], inplace=True)

# Find common keys
common_index = df_ground_truth.index.intersection(df.index)

# Filter and RESTORE index as columns
df_ground_truth_filtered = df_ground_truth.loc[common_index].reset_index()
df_filtered = df.loc[common_index].reset_index()

# Save with 合约代码 + 交易日期 columns included
df_ground_truth_filtered.to_csv("output_ground_truth_df.csv", index=False)
df_filtered.to_csv("output_mined_df.csv", index=False)
print(df_ground_truth_filtered.columns.values)
print(df_filtered.columns.values)
