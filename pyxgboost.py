import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from global_var import mapping_r2py, model_debug_dir, feature_cols
import itertools


# ------------------------------
# 数据准备（示例）
# ------------------------------
# 假设 df 是你的 DataFrame，包括特征和 label 列
train_data = pd.read_csv(f"{model_debug_dir}/2025-01-02py2train.csv")
verify_data = pd.read_csv(f"{model_debug_dir}/2025-01-02py2pred.csv")
train_data = train_data.rename(columns=mapping_r2py)

y = train_data["label"]
w = train_data["weight"]

X = train_data[feature_cols]
# X.to_csv("RVsPy/PyInputDataDropped.csv", index=False)
V = verify_data[feature_cols]
# print(V)

# train_rounds = [2500, 5000, 10000, 20000, 30000]
seeds = [1, 11, 111, 1111, 2024, 3333, 4444, 5555, 6666, 7777, 8888, 9999]
# process_rows = [0, 1, 2, 3, 4, 5]
# scale_rows = [0, 3, 5, 7, 9, 11, 13, 15, 17, 19]
# etas = [0.03, 0.05, 0.08, 0.1, 0.2]

train_rounds = [15000]
# seeds = [1031, 2222]
process_rows = [0]
scale_rows = [0]
shuffle_rows = [0]
etas = [0.03]
lambdas = [5]

result_buffer = []
# Cartesian product of train_round and seed
for (
    train_round,
    seed,
    shuffle_row,
    scale_row,
    drop_row,
    eta,
    lambda_,
) in itertools.product(
    train_rounds, seeds, shuffle_rows, scale_rows, process_rows, etas, lambdas
):

    X_local = X.copy()
    y_local = y.copy()
    w_local = w.copy()

    if drop_row > 0:
        X_local.drop(X_local.index[:drop_row], inplace=True)
        y_local.drop(y_local.index[:drop_row], inplace=True)
        w_local.drop(w_local.index[:drop_row], inplace=True)

    if scale_row > 0:
        # X_local.loc[X_local.index[:scale_row], "ROC_0"] *= 2
        w_local[:scale_row] *= 0.9

    if shuffle_row > 0:
        # shuffle the front shuffle_row rows
        X_local.iloc[:shuffle_row] = (
            X_local.iloc[:shuffle_row].sample(frac=1).reset_index(drop=True)
        )
        y_local.iloc[:shuffle_row] = (
            y_local.iloc[:shuffle_row].sample(frac=1).reset_index(drop=True)
        )
        w_local.iloc[:shuffle_row] = (
            w_local.iloc[:shuffle_row].sample(frac=1).reset_index(drop=True)
        )

    pos_w_all = w_local[y_local == 1].sum()
    neg_w_all = w_local[y_local == 0].sum()

    if pos_w_all > 0 and neg_w_all > 0:
        spw_all = min(neg_w_all / max(pos_w_all, 1), 100)
    else:
        spw_all = 1

    # train_rounds = [2500, 5000, 10000, 20000, 30000]
    # seeds = [1, 11, 111, 1111, 2024, 3333, 4444, 5555, 6666, 7777, 8888, 9999]

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": eta,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "lambda": lambda_,
        "alpha": 1,
        "scale_pos_weight": spw_all,
        "seed": seed,
    }
    dtrain = xgb.DMatrix(X_local.to_numpy(dtype=float), label=y_local, weight=w_local)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=train_round,
        evals=[(dtrain, "train")],
        verbose_eval=False,  # 对应 R 中的 verbose = 0
        early_stopping_rounds=100,
    )

    dverify = xgb.DMatrix(V)
    verify_preds = model.predict(dverify)
    print(
        "train_round: ",
        train_round,
        "seed: ",
        seed,
        "shuffle_rows: ",
        shuffle_row,
        "drop_rows: ",
        drop_row,
        "best_iter: ",
        model.best_iteration,
        "process_rows: ",
        scale_row,
        "eta: ",
        eta,
        "lambda: ",
        lambda_,
        "verify_preds: ",
        verify_preds,
    )
    result_buffer.append(
        {
            "train_round": train_round,
            "seed": seed,
            "best_iter": model.best_iteration,
            "shuffle_rows": shuffle_row,
            "drop_rows": drop_row,
            "process_rows": scale_row,
            "eta": eta,
            "lambda": lambda_,
            "verify_preds": verify_preds[0],
        }
    )

dump_xlsx = pd.DataFrame(
    result_buffer,
    columns=[
        "train_round",
        "seed",
        "best_iter",
        "shuffle_rows",
        "drop_rows",
        "process_rows",
        "eta",
        "lambda",
        "verify_preds",
    ],
)
dump_xlsx.to_excel("debug_pyxgboost.xlsx", index=False, engine="openpyxl")
