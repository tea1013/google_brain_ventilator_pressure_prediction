import sys

import numpy as np
from pandas.core.frame import DataFrame
from scipy import signal

sys.path.append("../../")

from typing import Callable, Dict, List, Optional

from teads.feature.feature_store import FeatureStore, feature

############################################## features ##############################################


@feature("base", "base", ["R", "C", "time_step", "u_in", "u_out"])
def feature_base(store: FeatureStore) -> Dict:
    d = {
        "R": store.target_df["R"],
        "C": store.target_df["C"],
        "time_step": store.target_df["time_step"],
        "u_in": store.target_df["u_in"],
        "u_out": store.target_df["u_out"],
    }

    return d


@feature("u_in", "u_in features", [])
def feature_u_in(store: FeatureStore) -> Dict:
    d = {}

    u_in_past_1 = store.target_df.groupby("breath_id")["u_in"].shift(1)
    u_in_past_2 = store.target_df.groupby("breath_id")["u_in"].shift(2)
    u_in_past_3 = store.target_df.groupby("breath_id")["u_in"].shift(3)
    u_in_past_4 = store.target_df.groupby("breath_id")["u_in"].shift(4)

    u_in_future_1 = store.target_df.groupby("breath_id")["u_in"].shift(-1)
    u_in_future_2 = store.target_df.groupby("breath_id")["u_in"].shift(-2)

    d["u_in_past_1"] = u_in_past_1
    d["u_in_past_2"] = u_in_past_2
    d["u_in_past_3"] = u_in_past_3
    d["u_in_past_4"] = u_in_past_4

    d["u_in_future_1"] = u_in_future_1
    d["u_in_future_2"] = u_in_future_2

    d["u_in_diff_1"] = store.target_df["u_in"] - u_in_past_1
    d["u_in_diff_2"] = store.target_df["u_in"] - u_in_past_2
    d["u_in_diff_3"] = store.target_df["u_in"] - u_in_past_3
    d["u_in_diff_4"] = store.target_df["u_in"] - u_in_past_4
    d["u_in_diff_-1"] = store.target_df["u_in"] - u_in_future_1
    d["u_in_diff_-2"] = store.target_df["u_in"] - u_in_future_2

    d["u_in_cumsum"] = store.target_df.groupby("breath_id")["u_in"].cumsum()

    d["u_in_first"] = store.target_df.groupby("breath_id")["u_in"].first()
    d["u_in_last"] = store.target_df.groupby("breath_id")["u_in"].last()

    d["u_in_diff_max"] = store.target_df.groupby("breath_id")["u_in"].agg("max") - store.target_df["u_in"]
    d["u_in_diff_first"] = store.target_df["u_in"] - d["u_in_first"]
    d["u_in_diff_last"] = store.target_df["u_in"] - d["u_in_last"]

    tmp_df = DataFrame()
    tmp_df["breath_id"] = store.target_df["breath_id"]
    tmp_df["u_in_sqrt"] = store.target_df["u_in"].apply(lambda x: np.sqrt(x))
    tmp_df["u_in_sqrt_past_1"] = tmp_df.groupby("breath_id")["u_in_sqrt"].shift(1)
    tmp_df["u_in_sqrt_past_2"] = tmp_df.groupby("breath_id")["u_in_sqrt"].shift(2)
    tmp_df["u_in_sqrt_past_3"] = tmp_df.groupby("breath_id")["u_in_sqrt"].shift(3)
    tmp_df["u_in_sqrt_past_4"] = tmp_df.groupby("breath_id")["u_in_sqrt"].shift(4)
    tmp_df["u_in_sqrt_future_1"] = tmp_df.groupby("breath_id")["u_in_sqrt"].shift(-1)
    tmp_df["u_in_sqrt_future_2"] = tmp_df.groupby("breath_id")["u_in_sqrt"].shift(-2)
    tmp_df["u_in_sqrt_cumsum"] = tmp_df.groupby("breath_id")["u_in_sqrt"].cumsum()

    d["u_in_sqrt"] = tmp_df["u_in_sqrt"]

    d["u_in_sqrt_past_1"] = tmp_df["u_in_sqrt_past_1"]
    d["u_in_sqrt_past_2"] = tmp_df["u_in_sqrt_past_2"]
    d["u_in_sqrt_past_3"] = tmp_df["u_in_sqrt_past_3"]
    d["u_in_sqrt_past_4"] = tmp_df["u_in_sqrt_past_4"]

    d["u_in_sqrt_future_1"] = tmp_df["u_in_sqrt_future_1"]
    d["u_in_sqrt_future_2"] = tmp_df["u_in_sqrt_future_2"]

    d["u_in_sqrt_diff_1"] = tmp_df["u_in_sqrt"] - tmp_df["u_in_sqrt_past_1"]
    d["u_in_sqrt_diff_2"] = tmp_df["u_in_sqrt"] - tmp_df["u_in_sqrt_past_2"]
    d["u_in_sqrt_diff_3"] = tmp_df["u_in_sqrt"] - tmp_df["u_in_sqrt_past_3"]
    d["u_in_sqrt_diff_4"] = tmp_df["u_in_sqrt"] - tmp_df["u_in_sqrt_past_4"]
    d["u_in_sqrt_diff_-1"] = tmp_df["u_in_sqrt"] - tmp_df["u_in_sqrt_future_1"]
    d["u_in_sqrt_diff_-2"] = tmp_df["u_in_sqrt"] - tmp_df["u_in_sqrt_future_2"]

    d["u_in_sqrt_cumsum"] = tmp_df["u_in_sqrt_cumsum"]

    d["u_in_diff_1_sign"] = np.sign(d["u_in_diff_1"])
    d["u_in_diff_2_sign"] = np.sign(d["u_in_diff_2"])
    d["u_in_diff_3_sign"] = np.sign(d["u_in_diff_3"])
    d["u_in_diff_4_sign"] = np.sign(d["u_in_diff_4"])

    d["u_in_sign_diff_1"] = d["u_in_diff_1_sign"] - d["u_in_diff_2_sign"]

    return d


@feature("u_out", "u_out features", [])
def feature_u_out(store: FeatureStore) -> Dict:
    d = {}

    d["u_out_past_1"] = store.target_df.groupby("breath_id")["u_out"].shift(1)
    d["u_out_past_2"] = store.target_df.groupby("breath_id")["u_out"].shift(2)
    d["u_out_past_3"] = store.target_df.groupby("breath_id")["u_out"].shift(3)
    d["u_out_past_4"] = store.target_df.groupby("breath_id")["u_out"].shift(4)
    d["u_out_future_1"] = store.target_df.groupby("breath_id")["u_out"].shift(-1)
    d["u_out_future_2"] = store.target_df.groupby("breath_id")["u_out"].shift(-2)

    return d


@feature("time_step", "time_step features", [])
def feature_time_step(store: FeatureStore) -> Dict:
    d = {}

    breath_time_df = DataFrame()
    breath_time_df["breath_id"] = store.target_df["breath_id"]
    breath_time_df["breath_time"] = store.target_df["time_step"] - store.target_df.groupby("breath_id")["time_step"].shift(1)

    d["breath_time"] = breath_time_df["breath_time"]
    d["last_time_step"] = store.target_df.groupby("breath_id")["time_step"].last()

    return d


@feature("RC", "R, C features", [])
def feature_RC(store: FeatureStore) -> Dict:
    d = {}

    R_map = {5: 0, 20: 1, 50: 2}
    C_map = {10: 0, 20: 1, 50: 2}
    R_C_map = {"5_10": 0, "5_20": 1, "5_50": 2, "20_10": 3, "20_20": 4, "20_50": 5, "50_10": 6, "50_20": 7, "50_50": 8}
    RC_sum_map = {v: i for i, v in enumerate([15, 25, 30, 40, 55, 60, 70, 100])}
    RC_mul_map = {v: i for i, v in enumerate([50, 100, 200, 250, 400, 500, 2500, 1000])}

    # R = store.target_df["R"]
    # R_cate = R.map(R_map)
    # d["R_cate"] = R_cate

    # C = store.target_df["C"]
    # C_cate = C.map(C_map)
    # d["C_cate"] = C_cate

    R_C = store.target_df["R"].astype(str) + "_" + store.target_df["C"].astype(str)
    R_C = R_C.map(R_C_map)
    d["R_C_cate"] = R_C

    # RC_sum = store.target_df["R"] + store.target_df["C"]
    # RC_sum_cate = RC_sum.map(RC_sum_map)
    # d["RC_sum_cate"] = RC_sum_cate

    # RC_mul = store.target_df["R"] * store.target_df["C"]
    # RC_mul_cate = RC_mul.map(RC_mul_map)
    # d["RC_mul_cate"] = RC_mul_cate

    return d


@feature("mix", "mix features", [])
def feature_mix(store: FeatureStore) -> Dict:
    d = {}

    tmp_df = DataFrame()
    tmp_df["breath_id"] = store.target_df["breath_id"]
    tmp_df["time_step"] = store.target_df["time_step"]
    tmp_df["u_in"] = store.target_df["u_in"]

    tmp_df["time_delta"] = tmp_df.groupby("breath_id")["time_step"].diff().fillna(0)
    tmp_df["delta"] = tmp_df["time_delta"] * tmp_df["u_in"]
    tmp_df["area"] = tmp_df.groupby("breath_id")["delta"].cumsum()

    d["delta"] = tmp_df["delta"]
    d["area"] = tmp_df["area"]

    return d


@feature("signal", "signal features", [])
def feature_signal(store: FeatureStore) -> Dict:
    def lowpass_filter(series, b, a):
        return signal.filtfilt(b, a, series)

    def get_agg_window(df: DataFrame, start_time: float = 0.0, end_time: float = 3.0, add_suffix: bool = False):
        df_tgt = df[(start_time <= df["time_step"]) & (df["time_step"] <= end_time)]
        df_feature = df_tgt.groupby(["breath_id"])["u_in"].apply(lowpass_filter, b=b, a=a)
        df_feature.name = "u_in_filter"

        return df_feature

    d = {}

    fp = 5  # 通過域端周波数[Hz]
    fs = 10  # 阻止域端周波数[Hz]
    gpass = 3  # 通過域端最大損失[dB]
    gstop = 40  # 阻止域端最小損失[dB]
    samplerate = 100

    fn = samplerate / 2  # ナイキスト周波数
    wp = fp / fn  # ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn  # ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  # オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "low")  # フィルタ伝達関数の分子と分母を計算

    df_agg_feature = get_agg_window(store.target_df).reset_index()
    df_agg_feature = df_agg_feature.explode("u_in_filter").reset_index(drop=True)
    df_agg_feature["u_in_filter"] = df_agg_feature["u_in_filter"].astype(float)
    df_agg_feature["u_in_filter_cumsum"] = df_agg_feature.groupby("breath_id")["u_in_filter"].cumsum()

    d["u_in_filter"] = df_agg_feature["u_in_filter"]
    d["u_in_filter_cumsum"] = df_agg_feature["u_in_filter_cumsum"]

    d["u_in_fiter_past_1"] = df_agg_feature.groupby("breath_id")["u_in_filter"].shift(1)
    d["u_in_fiter_past_2"] = df_agg_feature.groupby("breath_id")["u_in_filter"].shift(2)
    d["u_in_fiter_past_3"] = df_agg_feature.groupby("breath_id")["u_in_filter"].shift(3)
    d["u_in_fiter_past_4"] = df_agg_feature.groupby("breath_id")["u_in_filter"].shift(4)
    d["u_in_fiter_future_1"] = df_agg_feature.groupby("breath_id")["u_in_filter"].shift(-1)
    d["u_in_fiter_future_2"] = df_agg_feature.groupby("breath_id")["u_in_filter"].shift(-2)

    d["u_in_filter_diff_1"] = d["u_in_filter"] - d["u_in_fiter_past_1"]
    d["u_in_filter_diff_2"] = d["u_in_filter"] - d["u_in_fiter_past_2"]
    d["u_in_filter_diff_3"] = d["u_in_filter"] - d["u_in_fiter_past_3"]
    d["u_in_filter_diff_4"] = d["u_in_filter"] - d["u_in_fiter_past_4"]
    d["u_in_filter_diff_-1"] = d["u_in_filter"] - d["u_in_fiter_future_1"]
    d["u_in_filter_diff_-2"] = d["u_in_filter"] - d["u_in_fiter_future_2"]

    return d


############################################## Special features ##############################################


@feature("ids", "ids", ["id", "breath_id"])
def feature_id(store: FeatureStore) -> Dict:
    d = {"id": store.target_df["id"], "breath_id": store.target_df["breath_id"]}

    return d


############################################## features dict & transform ##############################################

f_dict = {
    "id": feature_id,
    "base": feature_base,
    "u_in": feature_u_in,
    "u_out": feature_u_out,
    "time_step": feature_time_step,
    "rc": feature_RC,
    "signal": feature_signal,
    "mix": feature_mix,
}


def get_feature_transforms(feat_ids: Optional[List[str]]) -> List[Callable]:
    return [f_dict[fid] for fid in feat_ids]
