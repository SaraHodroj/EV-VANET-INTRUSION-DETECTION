"""EV-VANET Intrusion Detection System: web API and app server.

Run locally:  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
import io
import json
import os
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sklearn.metrics import accuracy_score, f1_score

from .pipeline import CLASS_NAMES, FEATURES, REQUIRED, build_sequences

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
SAMPLES_DIR = ROOT / "samples"
STATIC_DIR = Path(__file__).resolve().parent / "static"
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "history.db"
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "50"))
MAX_ROWS_RETURNED = 5000

META = json.loads((MODEL_DIR / "model_meta.json").read_text())
MEAN = np.asarray(META["mean"], dtype=np.float32)
SCALE = np.asarray(META["scale"], dtype=np.float32)
MODEL = tf.keras.models.load_model(MODEL_DIR / "gru_ds2.keras")
EV_ATTACKS = {"Range Anxiety", "Fake RSU", "Charging DoS", "Billing Fraud"}

app = FastAPI(title="EV-VANET Intrusion Detection", version="1.0")


# ----------------------------------------------------------------------------- storage
def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


with closing(db()) as con:
    con.execute("""CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, filename TEXT,
        n_messages INTEGER, n_sequences INTEGER, n_alerts INTEGER, has_labels INTEGER,
        summary TEXT, result TEXT)""")
    con.commit()


def save(filename, result):
    s = result["summary"]
    with closing(db()) as con:
        cur = con.execute(
            "INSERT INTO analyses (created_at, filename, n_messages, n_sequences, n_alerts, has_labels, summary, result)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (result["created_at"], filename, s["messages"], s["sequences"], s["alerts"], int(result["has_labels"]),
             json.dumps(s), json.dumps(result)))
        con.commit()
        return cur.lastrowid


# ----------------------------------------------------------------------------- analysis
def analyse(df: pd.DataFrame, filename: str) -> dict:
    t0 = time.time()
    X, y, meta = build_sequences(df, with_labels=True)
    probs = MODEL.predict((X - MEAN) / SCALE, batch_size=4096, verbose=0)
    pred = probs.argmax(1)
    conf = probs.max(1)

    meta["predicted"] = [CLASS_NAMES[i] for i in pred]
    meta["confidence"] = np.round(conf, 4)
    meta["alert"] = pred != 0
    if y is not None:
        meta["actual"] = [CLASS_NAMES[i] if 0 <= i < len(CLASS_NAMES) else "Unknown" for i in y]

    counts = {c: int((pred == i).sum()) for i, c in enumerate(CLASS_NAMES)}
    n_alerts = int((pred != 0).sum())

    # one verdict per sender: the most frequent attack among its windows
    flagged = []
    for (city, sender), g in meta[meta["alert"]].groupby(["city", "sender_id"]):
        total = int((meta["sender_id"] == sender).sum())
        top = g["predicted"].value_counts()
        flagged.append({"city": city, "sender_id": sender, "attack": top.index[0],
                        "alert_windows": int(len(g)), "total_windows": total,
                        "share": round(len(g) / total, 3), "mean_confidence": round(float(g["confidence"].mean()), 3)})
    flagged.sort(key=lambda r: (-r["share"], -r["alert_windows"]))

    result = {
        "filename": filename,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "has_labels": y is not None,
        "summary": {
            "messages": int(len(df)), "sequences": int(len(pred)), "alerts": n_alerts,
            "alert_rate": round(n_alerts / len(pred), 4),
            "senders": int(meta.groupby(["city", "sender_id"]).ngroups),
            "flagged_senders": len(flagged),
            "ev_alerts": int(sum(counts[c] for c in EV_ATTACKS)),
            "seconds": round(time.time() - t0, 2),
        },
        "counts": counts,
        "flagged_senders": flagged[:500],
    }
    if y is not None:
        ok = (y >= 0) & (y < len(CLASS_NAMES))
        y, pred_e = y[ok], pred[ok]
    if y is not None and len(y):
        pred = pred_e
        benign = y == 0
        present = sorted(set(y.tolist()) | set(pred.tolist()))
        f1s = f1_score(y, pred, labels=present, average=None, zero_division=0)
        result["evaluation"] = {
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "macro_f1": round(float(f1_score(y, pred, labels=present, average="macro", zero_division=0)), 4),
            "false_alarm_rate": round(float((pred[benign] != 0).mean()), 4) if benign.any() else None,
            "attacks_detected": round(float((pred[~benign] != 0).mean()), 4) if (~benign).any() else None,
            "per_class_f1": {CLASS_NAMES[c]: round(float(f), 4) for c, f in zip(present, f1s) if 0 <= c < len(CLASS_NAMES)},
            "support": {CLASS_NAMES[c]: int((y == c).sum()) for c in present if 0 <= c < len(CLASS_NAMES)},
        }
    rows = meta.sort_values(["alert", "confidence"], ascending=[False, False]).head(MAX_ROWS_RETURNED)
    result["sequences"] = json.loads(rows.to_json(orient="records"))
    result["sequences_truncated"] = len(meta) > MAX_ROWS_RETURNED
    return result


def read_csv(raw: bytes) -> pd.DataFrame:
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB.")
    try:
        return pd.read_csv(io.BytesIO(raw))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not read the file as CSV: {e}")


def run(df, filename):
    try:
        result = analyse(df, filename)
    except ValueError as e:
        raise HTTPException(400, str(e))
    result["id"] = save(filename, result)
    return result


# ----------------------------------------------------------------------------- routes
@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    df = read_csv(await file.read())
    return run(df, file.filename or "upload.csv")


@app.get("/api/samples")
def samples():
    info = json.loads((SAMPLES_DIR / "samples.json").read_text())
    return info


@app.post("/api/samples/{name}/analyze")
def analyze_sample(name: str):
    path = SAMPLES_DIR / Path(name).name
    if not path.exists() or path.suffix != ".csv":
        raise HTTPException(404, "Sample not found.")
    return run(pd.read_csv(path), path.name)


@app.get("/api/samples/{name}")
def download_sample(name: str):
    path = SAMPLES_DIR / Path(name).name
    if not path.exists() or path.suffix != ".csv":
        raise HTTPException(404, "Sample not found.")
    return FileResponse(path, media_type="text/csv", filename=path.name)


@app.get("/api/template")
def template():
    cols = ["city", "sender_id", "timestamp_ms"] + FEATURES + ["attack_type"]
    return StreamingResponse(io.StringIO(",".join(cols) + "\n"), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=template.csv"})


@app.get("/api/history")
def history():
    with closing(db()) as con:
        rows = con.execute("SELECT id, created_at, filename, n_messages, n_sequences, n_alerts, has_labels, summary"
                           " FROM analyses ORDER BY id DESC LIMIT 200").fetchall()
    return [dict(r, summary=json.loads(r["summary"])) for r in rows]


@app.get("/api/history/{aid}")
def history_item(aid: int):
    with closing(db()) as con:
        r = con.execute("SELECT result FROM analyses WHERE id=?", (aid,)).fetchone()
    if not r:
        raise HTTPException(404, "Analysis not found.")
    res = json.loads(r["result"]); res["id"] = aid
    return res


@app.delete("/api/history/{aid}")
def delete_item(aid: int):
    with closing(db()) as con:
        con.execute("DELETE FROM analyses WHERE id=?", (aid,)); con.commit()
    return {"deleted": aid}


@app.get("/api/model")
def model_info():
    return {"architecture": "GRU (80 units, L1 0.001, dropout 0.2, dense 100)", "features": FEATURES,
            "required_columns": REQUIRED, "classes": CLASS_NAMES, "timesteps": META["timesteps"],
            "evaluation": META["evaluation"]}


@app.get("/api/health")
def health():
    return {"status": "ok"}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
