import pandas as pd
import argparse as arg

parser = arg.ArgumentParser()
parser.add_argument("--file", type=str, default="碳酸锂_predict_result.csv")
args = parser.parse_args()

df = pd.read_csv(args.file)

# if 预测值 or 实际值 is 无, then drop the row
df = df[df["预测值"] != "无"]
df = df[df["实际值"] != "无"]

# if 预测值 is 涨 and 实际值 is 涨, then accuracy + 1
# if 预测值 is 跌 and 实际值 is 跌, then accuracy + 1
# accuracy = (预测值为涨的次数 + 预测值为跌的次数) / 总行数
print(len(df[df["预测值"] == df["实际值"]]))
print(len(df))
accuracy = len(df[df["预测值"] == df["实际值"]]) / len(df)
print(f"准确率: {accuracy}")
