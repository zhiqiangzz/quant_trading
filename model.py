import numpy as np
import pandas as pd
import xgboost as xgb
import itertools
from dataclasses import dataclass, field
from typing import List
from global_var import model_debug_dir, mapping_py2r, feature_cols
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# ==================== 全局参数 ======================
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


@dataclass
class ModelParams:
    eta: List[float] = field(default_factory=lambda: [0.03])
    max_depth: List[int] = field(default_factory=lambda: [4])
    min_child_weight: List[int] = field(default_factory=lambda: [5])
    subsample: List[float] = field(default_factory=lambda: [0.75])
    colsample_bytree: List[float] = field(default_factory=lambda: [0.75])
    lambda_: List[float] = field(default_factory=lambda: [5])
    alpha: List[float] = field(default_factory=lambda: [1])
    seed: List[int] = field(
        default_factory=lambda: [
            1,
            11,
            111,
            1111,
            2024,
            3333,
            4444,
            5555,
            6666,
            7777,
            8888,
            9999,
        ]
    )
    boost_round: List[int] = field(default_factory=lambda: [15000])

    def get_product_params(self):
        return itertools.product(
            self.eta,
            self.max_depth,
            self.min_child_weight,
            self.subsample,
            self.colsample_bytree,
            self.lambda_,
            self.alpha,
            self.seed,
            self.boost_round,
        )


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
    df = df.sort_values(date_col).reset_index(drop=True)
    n_total = len(df)
    n_train_init = int(n_total * train_ratio)

    splits = []
    # 每次从当前窗口预测下一天
    train_start = n_total - n_train_init
    train_end = n_total

    for j in reversed(range(train_start, train_end)):
        if df.iloc[j]["IsPredicted"] == 1:
            val_day = j
            break

    train_df = df.iloc[train_start:train_end]
    val_df = df.iloc[[val_day]]
    splits.append((train_df, val_df))

    return splits


def compute_weights(
    df_sub, eps, main_alpha=MAIN_ALPHA, gamma=PRICE_GAMMA, cap_q=DIFF_CAP_Q
):
    # w_dead = pmin(1, abs(r_oc_next)/(eps+1e-8))
    w_dead = np.minimum(1, np.abs(df_sub["ROCNext"]) / (eps + 1e-8))

    # w_role：主力合约加权
    if "IsPredicted" in df_sub.columns:
        w_role = 1 + main_alpha * df_sub["IsPredicted"].isin([1, True]).astype(float)
    else:
        w_role = np.ones(len(df_sub))

    # w_px：基于 diff_abs_atr 的价格衰减
    if "DiffAbsAtr" in df_sub.columns:
        cap = np.quantile(df_sub["DiffAbsAtr"].dropna(), cap_q)
        if not np.isfinite(cap) or np.isnan(cap):
            cap = max(1.0, np.nanmedian(df_sub["DiffAbsAtr"]))

        diff_c = np.minimum(df_sub["DiffAbsAtr"], cap)
        w_px = np.exp(-gamma * diff_c)
    else:
        w_px = np.ones(len(df_sub))

    # final weight = w_dead * w_role * w_px, 并做 clipping
    w = w_dead * w_role * w_px
    w = np.clip(w, W_MIN, W_MAX)

    # # normalize
    w = w / np.nanmean(w)

    return w


# ==================== 主逻辑 ======================
def run_walkforward(
    train_set: pd.DataFrame, cut_off_date: pd.Timestamp, is_debug: bool = False
):

    # 1. 数据准备
    eps = estimate_eps(train_set["ROCNext"], train_set["ATRRelative_14"])
    label = make_labels_with_deadzone(train_set["ROCNext"], eps)
    keep = np.logical_not(np.isnan(label))
    train_set_kept = train_set.loc[keep]

    label = label[keep].astype(int)

    # 2. 权重计算
    w_train = compute_weights(train_set_kept, eps)

    max_date = train_set["交易日期"].max()
    age_win_all = (max_date - train_set["交易日期"]).dt.days

    age_win_f = age_win_all[keep]
    w_time_win = 0.5 ** (age_win_f / TIME_HALFLIFE)

    w_train = w_train * w_time_win
    w_train = np.clip(w_train, W_MIN, W_MAX)
    w_train = w_train / np.nanmean(w_train)

    # --- 修改开始 ---

    train_set_kept_feat = train_set_kept[feature_cols]
    X_full = train_set_kept_feat.to_numpy(dtype=float)

    y_full = label
    w_full = w_train

    if is_debug:
        dump_py2train = train_set_kept.copy()
        dump_py2train["label"] = y_full
        dump_py2train["weight"] = w_full
        dump_py2train.to_csv(
            f"{model_debug_dir}/{cut_off_date.strftime('%Y-%m-%d')}py2train.csv",
            index=False,
        )

    # 【关键修改点 1】：直接使用全量数据构建 dtrain
    dtrain = xgb.DMatrix(X_full, label=y_full, weight=w_full)

    # 基于全量数据计算 scale_pos_weight
    pos_w_all = w_full[y_full == 1].sum()
    neg_w_all = w_full[y_full == 0].sum()

    if pos_w_all > 0 and neg_w_all > 0:
        spw_all = min(neg_w_all / max(pos_w_all, 1), 100)
    else:
        spw_all = 1

    models = []
    hyper_params = ModelParams()

    for (
        eta_,
        max_depth_,
        min_child_weight_,
        subsample_,
        colsample_bytree_,
        lambda_,
        alpha_,
        seed_,
        boost_round_,
    ) in hyper_params.get_product_params():

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": eta_,
            "max_depth": max_depth_,
            "min_child_weight": min_child_weight_,
            "subsample": subsample_,
            "colsample_bytree": colsample_bytree_,
            "lambda": lambda_,
            "alpha": alpha_,
            "scale_pos_weight": spw_all,
            "seed": seed_,
        }

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=boost_round_,
            evals=[(dtrain, "train")],
            verbose_eval=False,  # 对应 R 中的 verbose = 0
            early_stopping_rounds=100,
        )

        m_win = len(X_full)

        if m_win >= 50:
            val_n = max(20, int(0.2 * m_win))

            X_calib = X_full[-val_n:]
            y_calib = y_full[-val_n:]

            dcalib = xgb.DMatrix(X_calib)
            p_val_raw = model.predict(dcalib)

            iso_reg = IsotonicRegression(
                y_min=0, y_max=1, out_of_bounds="clip", increasing=True
            )

            try:
                iso_reg.fit(p_val_raw, y_calib)
                model.iso_calibrator = iso_reg
            except Exception as e:
                print(f"Isotonic Regression failed: {e}")
                model.iso_calibrator = None
        else:
            model.iso_calibrator = None

        models.append(model)

    return models


def predict(
    xgb_models: List[xgb.Booster],
    predict_element: pd.DataFrame,
    cut_off_date: pd.Timestamp = None,
    is_debug: bool = False,
):

    if is_debug and cut_off_date is not None:
        predict_element.to_csv(
            f"{model_debug_dir}/{cut_off_date.strftime('%Y-%m-%d')}py2pred.csv",
            index=False,
        )

    raw_preds = (
        [
            xgb_model.predict(
                xgb.DMatrix(predict_element[feature_cols].to_numpy(dtype=float))
            )[0]
            for xgb_model in xgb_models
        ],
    )

    final_preds = [
        xgb_model.iso_calibrator.predict(raw_pred)[0]
        for xgb_model, raw_pred in zip(xgb_models, raw_preds)
    ]

    return np.mean(raw_preds), np.mean(final_preds)
