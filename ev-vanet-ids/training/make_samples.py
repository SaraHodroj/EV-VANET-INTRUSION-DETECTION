"""Build demo CSV files from the two cities held out of training (San Jose, Shanghai)."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from pipeline import FEATURES, CLASS_NAMES
SRC = sys.argv[1]
OUT = os.path.join(os.path.dirname(__file__), "..", "samples")
cols = ["city", "comm_type", "message_id", "sender_id", "receiver_id", "timestamp_ms"] + FEATURES + ["attack_type"]
df = pd.read_csv(SRC, usecols=cols)
df = df[df.city.isin(["san_jose", "shanghai"])]
lab = df.groupby("sender_id").attack_type.max()
rng = np.random.default_rng(7)

def senders(city, classes, n_per_class=None):
    s = lab[lab.index.str.startswith(city + "_")]
    out = []
    for c in classes:
        ids = s[s == c].index.to_numpy()
        if n_per_class and len(ids) > n_per_class:
            ids = rng.choice(ids, n_per_class, replace=False)
        out.extend(ids)
    return out

def write(name, ids, drop_label=False):
    d = df[df.sender_id.isin(ids)].sort_values(["sender_id", "timestamp_ms"])
    if drop_label:
        d = d.drop(columns=["attack_type"])
    d.to_csv(os.path.join(OUT, name), index=False, float_format="%.4f")
    return len(d), d.sender_id.nunique()

info = []
n, k = write("san_jose_mixed.csv", senders("san_jose", range(1, 11), 6) + senders("san_jose", [0], 120))
info.append({"file": "san_jose_mixed.csv", "title": "San Jose: mixed traffic", "description": "Benign vehicles and every attack type, from a city the model never saw", "messages": n, "senders": k, "labelled": True})
n, k = write("shanghai_ev_attacks.csv", senders("shanghai", [7, 8, 9, 10], 10) + senders("shanghai", [0], 80))
info.append({"file": "shanghai_ev_attacks.csv", "title": "Shanghai: EV-specific attacks", "description": "Fake RSU, Charging DoS, Billing Fraud and Range Anxiety among benign vehicles", "messages": n, "senders": k, "labelled": True})
n, k = write("shanghai_normal_only.csv", senders("shanghai", [0], 150))
info.append({"file": "shanghai_normal_only.csv", "title": "Shanghai: normal traffic only", "description": "Only benign vehicles, to show how rarely the detector raises false alarms", "messages": n, "senders": k, "labelled": True})
n, k = write("san_jose_unlabelled.csv", senders("san_jose", [1, 6, 8, 9], 4) + senders("san_jose", [0], 60), drop_label=True)
info.append({"file": "san_jose_unlabelled.csv", "title": "San Jose: unlabelled", "description": "No attack labels, as real traffic would arrive; the app only reports detections", "messages": n, "senders": k, "labelled": False})
json.dump(info, open(os.path.join(OUT, "samples.json"), "w"), indent=1)
print(json.dumps(info, indent=1))
