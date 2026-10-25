"""Sequence construction shared by training and the web app (thesis Section 3.6)."""
import numpy as np
import pandas as pd

FEATURES = ["hop_count", "end_to_end_delay_ms", "inter_arrival_ms", "message_size_bytes", "rssi_dbm",
            "distance_m", "packet_loss_rate", "burst_count_per_sec", "reported_x", "reported_y", "speed_ms",
            "acceleration_ms2", "heading_deg", "battery_level_pct", "signal_phase", "charging_power_kw",
            "billing_amount"]
CLASS_NAMES = ["Benign", "DoS", "Sybil", "Position Falsification", "Replay", "Bogus Information",
               "Black Hole", "Range Anxiety", "Fake RSU", "Charging DoS", "Billing Fraud"]
REQUIRED = ["sender_id", "timestamp_ms"] + FEATURES
TIMESTEPS = 3


def build_sequences(df: pd.DataFrame, with_labels: bool = False):
    """Group messages by sender, order by time, cut non-overlapping windows of 3.

    Returns X (n, 3, 17), y (or None) and a metadata frame (one row per sequence)
    describing the last message of each window.
    """
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    df = df.copy()
    if "city" not in df.columns:
        df["city"] = "uploaded"
    df = df.sort_values(["city", "sender_id", "timestamp_ms"], kind="mergesort").reset_index(drop=True)
    feats = df[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    codes = df.groupby(["city", "sender_id"], sort=False).ngroup().to_numpy()
    bounds = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [len(df)]])
    idx = []
    for s, e in zip(starts, ends):
        n = e - s
        if n < TIMESTEPS:
            continue
        idx.extend(range(s, s + (n // TIMESTEPS) * TIMESTEPS, TIMESTEPS))
    if not idx:
        raise ValueError("No sender has at least 3 messages, so no sequence can be formed.")
    idx = np.asarray(idx)
    gather = idx[:, None] + np.arange(TIMESTEPS)[None, :]
    X = feats[gather]
    last = df.iloc[gather[:, -1]]
    keep = [c for c in ["city", "sender_id", "timestamp_ms", "comm_type", "message_id"] if c in df.columns]
    meta = last[keep].reset_index(drop=True)
    meta["messages_in_window"] = TIMESTEPS
    y = None
    if with_labels and "attack_type" in df.columns:
        lab = pd.to_numeric(df["attack_type"], errors="coerce").fillna(-1).to_numpy().astype(np.int32)
        y = lab[gather[:, -1]]
    return X, y, meta
