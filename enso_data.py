"""C2 data pipeline (Annex D3). Month index m = 0 is January 1950; m = 899 is December 2024."""
import numpy as np, pandas as pd

LEADS = [1, 3, 6, 9, 12]
HIST = 24
N_MONTHS = 900
TRAIN_END = (1994 - 1950) * 12 + 11   # 539  = Dec 1994
VAL_END = (2004 - 1950) * 12 + 11     # 659  = Dec 2004
PART_NAMES = ["train", "val", "test"]


def part_of_month(m):
    return 0 if m <= TRAIN_END else (1 if m <= VAL_END else 2)


def label(m):
    return f"{1950 + m // 12}-{m % 12 + 1:02d}"


def load_raw(ascii_path, npz_path):
    df = pd.read_csv(ascii_path, sep=r"\s+")
    df.columns = ["YR", "MON", "N12", "N12A", "N3", "N3A", "N4", "N4A", "N34", "N34A"]
    s = df[(df.YR >= 1950) & (df.YR <= 2024)].reset_index(drop=True)
    assert len(s) == N_MONTHS
    z = np.load(npz_path)
    assert (z["year"] == s.YR.values).all() and (z["month"] == s.MON.values).all()
    return s.N34A.values.astype(np.float64), z["sst"].astype(np.float32), s.MON.values


def split_table():
    """One row per possible issue month with >=24 months of history and a 12-month-ahead target."""
    rows = []
    for i in range(HIST - 1, N_MONTHS - max(LEADS)):
        tparts = {part_of_month(i + h) for h in LEADS}
        keep = len(tparts) == 1
        rows.append(dict(issue_idx=i, issue=label(i), first_target=label(i + 1), last_target=label(i + 12),
                         partition=PART_NAMES[tparts.pop()] if keep else "excluded"))
    return pd.DataFrame(rows)


def build(ascii_path, npz_path, include_test_labels=False):
    """Returns dict of inputs/targets. Test labels are NOT materialised unless asked for."""
    y, sst, mon = load_raw(ascii_path, npz_path)
    tab = split_table()
    idx = {p: tab.issue_idx[tab.partition == p].values for p in PART_NAMES}

    # Grid anomalies: monthly climatology from 1950-1994 only
    clim = np.stack([sst[:TRAIN_END + 1][mon[:TRAIN_END + 1] == k].mean(0) for k in range(1, 13)])
    anom = sst - clim[mon - 1]
    ocean = ~np.isnan(anom[0])

    def win(ii):  # (n, 24, 21, 81) oldest -> newest
        return np.stack([anom[i - HIST + 1:i + 1] for i in ii])

    Xtr = win(idx["train"])
    mu = np.nanmean(Xtr, axis=(0, 2, 3), keepdims=True)[0]   # per-channel, training examples only, ocean cells only
    sd = np.nanstd(Xtr, axis=(0, 2, 3), keepdims=True)[0]

    def prep_map(X):
        X = (X - mu) / sd
        return np.nan_to_num(X, nan=0.0).astype(np.float32)

    # Index inputs: standardised with training-example statistics; targets stay in degC
    def idx_win(ii):
        return np.stack([y[i - HIST + 1:i + 1] for i in ii])
    itr = idx_win(idx["train"])
    imu, isd = itr.mean(), itr.std()

    def prep_idx(X):
        return ((X - imu) / isd).astype(np.float32)[..., None]

    def targets(ii):
        return np.stack([[y[i + h] for h in LEADS] for i in ii]).astype(np.float32)

    out = dict(table=tab, idx=idx, ocean=ocean, stats=dict(chan_mean=mu, chan_std=sd, idx_mean=imu, idx_std=isd))
    for p in PART_NAMES:
        out[f"Xmap_{p}"] = prep_map(Xtr if p == "train" else win(idx[p]))
        out[f"Xidx_{p}"] = prep_idx(idx_win(idx[p]))
        out[f"Xraw_{p}"] = idx_win(idx[p]).astype(np.float32)   # degC lags, for persistence / ridge
        if p != "test" or include_test_labels:
            out[f"Y_{p}"] = targets(idx[p])
    return out
