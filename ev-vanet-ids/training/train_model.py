"""Train the deployable GRU detector with the thesis configuration.

Configuration (thesis Sections 3.6-3.7): 17 traffic features, sequences of 3
consecutive messages from one sender, StandardScaler fitted on training cities,
no class weighting, GRU(80) with L1 = 0.001, dropout 0.2, Dense(100), Adam,
batch size 256, up to 100 epochs, early stopping (patience 10) on validation loss,
learning rate halved after 5 epochs without improvement.

Two cities (San Jose, Shanghai) are held out completely, so the app can be
demonstrated on road networks the model has never seen.
"""
import json, os, sys, time
import numpy as np, pandas as pd
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from pipeline import FEATURES, CLASS_NAMES, build_sequences  # same code the app uses

SRC = sys.argv[1]
OUT = os.path.join(os.path.dirname(__file__), "..", "model")
TEST_CITIES = ["san_jose", "shanghai"]
VAL_CITY = "oslo"
SEED = 42

import tensorflow as tf
tf.keras.utils.set_random_seed(SEED)
from tensorflow.keras import layers, regularizers, callbacks
from sklearn.metrics import f1_score, accuracy_score

df = pd.read_csv(SRC)
X, y, meta = build_sequences(df, with_labels=True)
city = meta["city"].values
tr = ~np.isin(city, TEST_CITIES + [VAL_CITY]); va = city == VAL_CITY; te = np.isin(city, TEST_CITIES)
mean = X[tr].reshape(-1, X.shape[2]).mean(0); scale = X[tr].reshape(-1, X.shape[2]).std(0); scale[scale == 0] = 1.0
Xs = (X - mean) / scale
print("sequences", len(y), "train", tr.sum(), "val", va.sum(), "test", te.sum(), flush=True)

l1 = regularizers.l1(0.001)
model = tf.keras.Sequential([
    layers.Input(shape=(3, len(FEATURES))),
    layers.GRU(80, kernel_regularizer=l1, recurrent_regularizer=l1, bias_regularizer=l1),
    layers.Dropout(0.2), layers.Dense(100, activation="relu"),
    layers.Dense(len(CLASS_NAMES), activation="softmax")])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
t0 = time.time()
model.fit(Xs[tr], y[tr], validation_data=(Xs[va], y[va]), epochs=100, batch_size=256, verbose=2,
          callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                     callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)])
p = model.predict(Xs[te], batch_size=4096, verbose=0).argmax(1)
benign = y[te] == 0
res = {"test_cities": TEST_CITIES, "validation_city": VAL_CITY,
       "accuracy": float(accuracy_score(y[te], p)), "macro_f1": float(f1_score(y[te], p, average="macro")),
       "false_alarm_rate": float((p[benign] != 0).mean()), "attacks_detected": float((p[~benign] != 0).mean()),
       "per_class_f1": dict(zip(CLASS_NAMES, map(float, f1_score(y[te], p, labels=range(len(CLASS_NAMES)), average=None)))),
       "train_seconds": round(time.time() - t0)}
print(json.dumps(res, indent=1), flush=True)
os.makedirs(OUT, exist_ok=True)
model.save(os.path.join(OUT, "gru_ds2.keras"))
json.dump({"features": FEATURES, "class_names": CLASS_NAMES, "mean": mean.tolist(), "scale": scale.tolist(),
           "timesteps": 3, "evaluation": res}, open(os.path.join(OUT, "model_meta.json"), "w"), indent=1)
print("saved", flush=True)
