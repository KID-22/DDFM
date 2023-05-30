import copy
import pandas as pd
import numpy as np
from tqdm import tqdm

SECONDS_A_MINUTE = 60
SECONDS_FIVE_MINUTE = SECONDS_A_MINUTE * 5
SECONDS_TEN_MINUTE = SECONDS_A_MINUTE * 10
SECONDS_A_QUATER = SECONDS_A_MINUTE * 15
SECONDS_TWO_QUATER = SECONDS_A_QUATER * 2
SECONDS_AN_HOUR = SECONDS_A_MINUTE * 60
SECONDS_SIX_HOUR = SECONDS_AN_HOUR * 6
SECONDS_HALF_A_DAY = SECONDS_AN_HOUR * 12
SECONDS_A_DAY = SECONDS_AN_HOUR * 24
SECONDS_THREE_DAY = SECONDS_A_DAY * 3
SECONDS_FIVE_DAY = SECONDS_A_DAY * 5
SECONDS_SENVEN_DAY = SECONDS_A_DAY * 7
SECONDS_NINE_DAY = SECONDS_A_DAY * 9
SECONDS_HALF_MONTH = SECONDS_A_DAY * 15
SECONDS_A_MONTH = SECONDS_A_DAY * 30

time_dict = {
    "SECONDS_A_MINUTE": SECONDS_A_MINUTE,
    "SECONDS_FIVE_MINUTE": SECONDS_FIVE_MINUTE,
    "SECONDS_TEN_MINUTE": SECONDS_TEN_MINUTE,
    "SECONDS_A_QUATER": SECONDS_A_QUATER,
    "SECONDS_TWO_QUATER": SECONDS_TWO_QUATER,
    "SECONDS_AN_HOUR": SECONDS_AN_HOUR,
    "SECONDS_SIX_HOUR": SECONDS_SIX_HOUR,
    "SECONDS_HALF_A_DAY": SECONDS_HALF_A_DAY,
    "SECONDS_A_DAY":SECONDS_A_DAY,
    "SECONDS_THREE_DAY": SECONDS_THREE_DAY,
    "SECONDS_FIVE_DAY": SECONDS_FIVE_DAY,
    "SECONDS_SENVEN_DAY": SECONDS_SENVEN_DAY,
    "SECONDS_HALF_MONTH": SECONDS_HALF_MONTH,
    "SECONDS_A_MONTH": SECONDS_A_MONTH,
    "SECONDS_FIVE_DAY": SECONDS_FIVE_DAY,
    "SECONDS_SENVEN_DAY": SECONDS_SENVEN_DAY,
    "SECONDS_NINE_DAY": SECONDS_NINE_DAY
}


SECONDS_DELAY_NORM = 1
SECONDS_FSIW_NORM = SECONDS_A_DAY*5

num_bin_size = (64, 16, 128, 64, 128, 64, 512, 512)
cate_bin_size = (512, 128, 256, 256, 64, 256, 256, 16, 256)

def get_criteo_data_df(params):
    df = pd.read_csv(params["data_path"], sep="\t", header=None)
    click_ts = df[df.columns[0]].to_numpy()
    pay_ts = df[df.columns[1]].fillna(-1).to_numpy()

    df = df[df.columns[2:]]
    for c in df.columns[8:]:
        df[c] = df[c].fillna("")
        df[c] = df[c].astype(str)
    for c in df.columns[:8]:
        df[c] = df[c].fillna(-1)
        df[c] = (df[c] - df[c].min())/(df[c].max() - df[c].min())
    df.columns = [str(i) for i in range(17)]
    df.reset_index(inplace=True)
    data = []
    for i in range(8, 17):
        c = str(i)
        hash_value = pd.util.hash_array(df[c].to_numpy()) % cate_bin_size[i-8]
        data.append(hash_value.reshape(-1, 1))
    for i in range(8):
        c = str(i)
        labels = list(range(num_bin_size[i]))
        out, bins = pd.cut(df[c], bins=num_bin_size[i], retbins=True, labels=labels)
        data.append(out.to_numpy().reshape((-1, 1)))
    data = np.concatenate(data, axis=1)
    return data, click_ts, pay_ts