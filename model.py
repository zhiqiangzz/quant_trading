import numpy as np
import pandas as pd
import xgboost as xgb
import math
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import confusion_matrix, accuracy_score

# ==================== 全局参数 ======================
UP_TH = 0.60
DN_TH = 0.40
EPS_ALPHA = 0.20
EPS_K = 0.25

MAIN_ALPHA = 0.80
PRICE_GAMMA = 0.90
DIFF_CAP_Q = 0.98
W_MIN, W_MAX = 0.20, 4.00

ISO_MIN_UNIQUE = 10
ISO_MIN_POS = 20
ISO_MIN_NEG = 20
ISO_MAX_ZERO_SP = 0.50
PLATT_FALLBACK = True
PLATT_L2 = 1.0
P_EPS = 1e-4
TIME_HALFLIFE = 30


# ==================== 工具函数 ======================
def estimate_eps(r_next, atr_rel=None, alpha=EPS_ALPHA, k=EPS_K):
    """估计 dead-zone 阈值"""
    q_abs = np.nanquantile(np.abs(r_next), alpha)
    atr_term = 0
    if atr_rel is not None:
        atr_term = k * np.nanmedian(np.abs(atr_rel))
    return max(q_abs, atr_term)


def make_labels_with_deadzone(r_next, eps):
    """生成涨跌标签 (1,0,NaN)"""
    y = np.where(r_next > eps, 1, np.where(r_next < -eps, 0, np.nan))
    return y


def compute_weights(df, eps, w_time=None):
    """样本加权"""
    w_dead = np.minimum(1, np.abs(df["ROCNext"]) / (eps + 1e-8))
    w_role = 1 + MAIN_ALPHA * df.get("IsMain", 0)
    if "DiffAbsATR14" in df:
        cap = np.nanquantile(df["DiffAbsATR14"], DIFF_CAP_Q)
        diff_c = np.minimum(df["DiffAbsATR14"], cap)
        w_px = np.exp(-PRICE_GAMMA * diff_c)
    else:
        w_px = 1
    w = w_dead * w_role * w_px * (w_time if w_time is not None else 1)
    w = np.clip(w, W_MIN, W_MAX)
    return w / np.nanmean(w)


def should_fallback_platt(p_raw, y_bin):
    uniq = np.unique(np.round(p_raw, 6))
    zero_span_ratio = np.mean(np.diff(np.sort(p_raw)) == 0)
    pos_n = np.sum(y_bin == 1)
    neg_n = np.sum(y_bin == 0)
    return (
        len(uniq) < ISO_MIN_UNIQUE
        or pos_n < ISO_MIN_POS
        or neg_n < ISO_MIN_NEG
        or zero_span_ratio > ISO_MAX_ZERO_SP
    )


def calibrate(p_raw, y_bin):
    """拟合概率校准器"""
    if should_fallback_platt(p_raw, y_bin) and PLATT_FALLBACK:
        lr = LogisticRegression(C=1 / PLATT_L2, solver="lbfgs")
        lr.fit(p_raw.reshape(-1, 1), y_bin)
        return ("platt", lr)
    else:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_raw, y_bin)
        return ("iso", iso)


def predict_calibrated(calibrator, p_raw):
    kind, model = calibrator
    if kind == "iso":
        return model.predict(p_raw)
    elif kind == "platt":
        return model.predict_proba(p_raw.reshape(-1, 1))[:, 1]
    else:
        return p_raw


def walk_forward_split(df, date_col="交易日期", train_ratio=0.6):
    """
    对时间序列 df 进行 walk-forward 拆分：
    - 初始窗口：前 train_ratio 比例的数据；
    - 每次训练集向前滚动 1 天；
    - 每次验证集为下一天；
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    n_total = len(df)
    n_train_init = int(n_total * train_ratio)

    splits = []
    # 每次从当前窗口预测下一天
    for i in range(n_train_init, n_total):
        train_start = i - n_train_init
        train_end = i  # 不含第 i 行
        if df.iloc[i]["交易日期"] == df.iloc[i - 1]["交易日期"]:
            # find the next day that is not the same as the current day
            for j in range(i + 1, n_total):
                if df.iloc[j]["交易日期"] != df.iloc[i]["交易日期"]:
                    train_end = j
                    break

        val_day = i  # 用于验证
        for j in range(train_end, n_total):
            if df.iloc[j]["IsMain"] == 1:
                val_day = j

        train_df = df.iloc[train_start:train_end]
        val_df = df.iloc[[val_day]]  # 验证1天
        splits.append((train_df, val_df))

    return splits


# ==================== 主逻辑 ======================
def run_walkforward(df: pd.DataFrame):
    df = df.sort_values("交易日期").reset_index(drop=True)
    feat_cols = [
        c
        for c in df.columns
        if c
        not in ["交易日期", "合约代码", "品种名称", "ROCNext", "IsMain", "DiffAbsATR14"]
    ]

    preds, trues = [], []

    for train_set, test_set in walk_forward_split(df):
        eps = estimate_eps(train_set["ROCNext"], train_set["ATRRel14"])
        label = make_labels_with_deadzone(train_set["ROCNext"], eps)
        keep = np.logical_not(np.isnan(label))
        train_set = train_set.loc[keep]
        label = label[keep].astype(int)
        tr_label = label[:split_point]
        val_label = label[split_point:]

        max_date = w_train["交易日期"].max()
        age_all = (max_date - w_train["交易日期"]).dt.days.astype(float)
        w_time_all = 0.5 ** (age_all / TIME_HALFLIFE)

        total_size = len(train_set)

        split_point = max(20, math.floor(0.15 * total_size))

        if total_size <= split_point + 5:
            split_point = max(5, math.floor(0.1 * total_size))

        total_data = train_set[feat_cols].to_numpy(dtype=float)
        data_to_train = total_data[:split_point]
        data_to_val = total_data[split_point:]

        w_train = compute_weights(train_set, eps, w_time_all)
        w_train_tr = w_train[:split_point]
        w_train_val = w_train[split_point:]

        pos_w = w_train_tr[tr_label == 1].sum()
        neg_w = w_train_tr[tr_label == 0].sum()

        if pos_w > 0 and neg_w > 0:
            spw = min(neg_w / max(pos_w, 1), 100)
        else:
            spw = 1

        # XGBoost 训练
        dtrain = xgb.DMatrix(data_to_train, label=tr_label, weight=w_train_tr)
        dval = xgb.DMatrix(data_to_val, label=val_label, weight=w_train_val)
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": 0.05,
            "max_depth": 4,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "lambda": 2,
            "alpha": 1,
            "scale_pos_weight": spw,
        }

        model = xgb.train(
            params,
            evals=[(dtrain, "train"), (dval, "eval")],
            num_boost_round=300,
            early_stopping_rounds=100,
        )

        # 校准
        p_val_raw = model.predict(dtrain)
        kind, calib = calibrate(p_val_raw, label)

        X_test = test_set[feat_cols].to_numpy(dtype=float)
        p_test_raw = model.predict(xgb.DMatrix(X_test))
        p_test = predict_calibrated((kind, calib), p_test_raw)
        p_test = np.clip(p_test, P_EPS, 1 - P_EPS)

        pred = np.where(p_test >= UP_TH, 1, np.where(p_test <= DN_TH, 0, np.nan))
        true = make_labels_with_deadzone(test_set["ROCNext"], eps)
        preds.append(pred[0])
        trues.append(true[0])

    preds, trues = np.array(preds), np.array(trues)
    mask = ~np.isnan(preds) & ~np.isnan(trues)
    if np.sum(mask) == 0:
        print("可交易样本过少")
        return None
    acc = accuracy_score(trues[mask], preds[mask])
    cm = confusion_matrix(trues[mask], preds[mask])
    print("Confusion matrix:\n", cm)
    print(f"Walk-forward Accuracy={acc:.4f} | Trade rate={100*np.mean(mask):.2f}%")
    return acc, model, (kind, calib), feat_cols


# ==================== 预测接口 ======================
def predict_next(model, calibrator, feat_cols, next_df):
    X_next = next_df[feat_cols].to_numpy(dtype=float)
    p_raw = model.predict(xgb.DMatrix(X_next))
    p = predict_calibrated(calibrator, p_raw)
    p = np.clip(p, P_EPS, 1 - P_EPS)
    cls = np.where(p >= UP_TH, "涨", np.where(p <= DN_TH, "跌", "无"))
    return pd.DataFrame({"prob": p, "class": cls})
