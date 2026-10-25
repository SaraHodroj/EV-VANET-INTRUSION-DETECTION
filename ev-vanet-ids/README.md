# EV-VANET Intrusion Detection – Web App

A web application for the GRU intrusion detector from the thesis *Intrusion Detection in Automotive Communication Networks (VANET) for Electric Vehicles* (Sara Hodroj, Lebanese University).

Upload a CSV of vehicle messages, and the app groups them by sender, builds sequences of three consecutive messages, classifies each sequence into benign or one of ten attack types, and shows:

- a summary: messages, sequences, alerts and flagged senders
- detections by attack type, with the four EV-specific attacks highlighted
- the flagged senders and their main attack
- every sequence with its detected class and confidence, downloadable as CSV
- accuracy, macro F1, false alarm rate and attacks detected, when the file contains labels (`attack_type`)
- a saved history of all analyses

Four sample files from **San Jose and Shanghai**, two cities the model never saw in training, are included.

## Model

| | |
|---|---|
| Architecture | GRU (80 units, L1 0.001, dropout 0.2), Dense 100, softmax over 11 classes |
| Input | 3 consecutive messages × 17 traffic features, standard-scaled |
| Training data | Dense urban dataset (DS2), 7 cities for training, Oslo for validation |
| Held-out test | San Jose and Shanghai: accuracy 97.1%, macro F1 0.848, 91.0% of attacks detected, 0.05% false alarms |

Retrain with `python training/train_model.py <dense_urban_enriched.csv>` and rebuild the samples with `python training/make_samples.py <same file>`.

## Input format

One row per message. Download the template from the app (**Download CSV template**).

- Required: `sender_id`, `timestamp_ms` and the 17 features: `hop_count, end_to_end_delay_ms, inter_arrival_ms, message_size_bytes, rssi_dbm, distance_m, packet_loss_rate, burst_count_per_sec, reported_x, reported_y, speed_ms, acceleration_ms2, heading_deg, battery_level_pct, signal_phase, charging_power_kw, billing_amount`
- Optional: `city`, `comm_type`, `message_id`, and `attack_type` (0 = benign … 10 = Billing Fraud) to evaluate the detector
- Senders with fewer than 3 messages are skipped, as in the thesis

## Run on your computer

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

Then open http://localhost:8000.

With Docker:

```bash
docker build -t ev-vanet-ids .
docker run -p 8000:8000 -v ev-ids-data:/data ev-vanet-ids
```

## Put it online

### Option A: Render (free tier)

1. Create a GitHub repository and upload the contents of this folder.
2. On https://render.com, choose **New → Web Service**, connect the repository.
3. Render detects the `Dockerfile`. Choose the free instance type and click **Create Web Service**.
4. After the build (about 5–10 minutes), the app is live at `https://<name>.onrender.com`.

The free tier sleeps after 15 minutes without visits; the first visit afterwards takes about a minute. Its disk is not permanent, so the history is cleared when the service restarts; add a Render disk mounted at `/data` to keep it.

### Option B: Hugging Face Spaces (free)

1. On https://huggingface.co/new-space, choose **Docker** as the SDK.
2. Upload the contents of this folder.
3. Add these lines at the very top of this README (Spaces reads them):

```
---
title: EV-VANET Intrusion Detection
sdk: docker
app_port: 7860
---
```

4. In the Space settings, add the variable `PORT` = `7860`. The Space builds and starts automatically.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/analyze` | Analyse an uploaded CSV (`file` form field) |
| GET | `/api/samples` | List sample files |
| POST | `/api/samples/{file}/analyze` | Analyse a sample |
| GET | `/api/history` · `/api/history/{id}` | List or open saved analyses |
| DELETE | `/api/history/{id}` | Delete an analysis |
| GET | `/api/model` | Model description and held-out evaluation |
| GET | `/api/template` | Empty CSV with the expected columns |

Interactive API documentation is at `/docs`.

## Limitations

The detector follows the thesis: it works on message-level features, so position falsification, which changes no observable feature, is not detected. It was trained on simulated data (SUMO), so performance on real V2X traffic has not been measured.
