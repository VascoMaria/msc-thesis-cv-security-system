# Security System

This directory contains the core implementation of the **Security System** developed as part of **my Master's Thesis**.

The system follows a **modular and distributed architecture** for real-time security monitoring in a self-service machine environment. It integrates multiple computer vision inference services and configurable decision logic to detect potentially dangerous situations.

---

## System Components

- **API (`api/`)**  
  Entry point of the system. Receives image frames from the self-service machine camera (or a test client), validates them, and forwards them for processing.

- **Controller (`controller/`)**  
  Orchestrates the distributed inference services, aggregates results, and applies decision rules to produce the final alarm decision.

- **Inference Engine (`inference_engine/`)**  
  A collection of independent **model services** (one service per model) that expose HTTP endpoints (e.g., `/detect_batch`).

- **Common (`common/`)**  
  Shared configuration, decision rules, thresholds, weights, and utility functions.

---

## How to Run the Full System (End-to-End)

The system is designed to run as **multiple services**. The typical startup order is:

### 1) Start the inference services (one per model)
Start each model service inside `inference_engine/` on the ports defined in `common/config.json` (e.g., 8000, 8001, 8006, 8007, 8009, 8010, 8012).  
Each service must expose the configured endpoint (commonly `/detect_batch`). 

### 2) Configure the system (`common/config.json`)
Before starting the API, configure:
- Which categories are active (`active: true/false`)
- The URL/port of each model service (`url`)
- Decision rules and thresholds (global + per-category)
- Weights and alarm classes

See **Configuration** below for details. 

### 3) Start the API service
Run the API (which will import and use the Controller and Common modules internally):

```bash
# from the API folder
python api.py
```

or with Uvicorn (development mode):

```bash
uvicorn api:app --host 0.0.0.0 --port 8050 --reload
```

### 4) Send frames to the API (client/test)
Send frames from a client to the API endpoint (`POST /process_frame`).  
The API accepts **one frame per request**, but buffers frames and processes them internally in batch mode.

Example (multiple images sent sequentially):

**Linux/macOS (bash):**
```bash
for img in frame1.jpg frame2.jpg frame3.jpg; do
  curl -X POST "http://127.0.0.1:8050/process_frame"        -H "Content-Type: multipart/form-data"        -F "frame=@$img"
done
```

**Windows (PowerShell):**
```powershell
$images = @("frame1.jpg", "frame2.jpg", "frame3.jpg")
foreach ($img in $images) {
    curl -X POST "http://127.0.0.1:8050/process_frame" `
         -H "Content-Type: multipart/form-data" `
         -F "frame=@$img"
}
```

---

## Buffer Sizing Guidance (API)

The API uses an internal buffer to group incoming frames before processing.

The buffer size should be configured according to the average processing time of one inference cycle multiplied by the frame rate (FPS) of the client, in order to avoid frame loss and ensure stable real-time performance.

General guideline:

```
buffer_size ≈ processing_time_per_cycle × client_FPS
```

---

## Configuration (`common/config.json`)

System behavior is driven by `common/config.json`. 

### Global decision settings
- `categories_decision_rule`: how alarms are combined across categories (e.g., `scoring`, `consensus`)
- `threshold`: global threshold (used for global scoring rules)

### Per-category settings
Each category includes:
- `name`: category identifier (e.g., `weapon_detection`, `emotion_recognition`, `environment_detection`)
- `active`: enable/disable the category
- `decision_rule`: decision method for the category (e.g., `scoring`, `consensus`, `prioridade`)
- `threshold`: category threshold (when applicable)
- `weight`: category weight used in global scoring rules
- `models`: list of model services for that category

### Per-model settings
Each model includes:
- `name`: model identifier
- `url`: full service URL (including port and endpoint)
- `classes`: alarm-relevant classes (and optional class weights)
- `thresholdDetection`: minimum confidence used to consider a detection
- `weight`: model weight used by category decision rules

---

## Endpoints (API)

- `GET /` — health check
- `POST /process_frame` — submit a single frame for processing (buffered internally)
- `GET /status` — last alarm information (if any)

For details, see `api/README.md`.

---

## Notes

- The API does **not** run inference directly; it forwards processing to the Controller.
- The Controller dispatches requests to each active model service asynchronously.
- Ensure all configured model services are reachable at the URLs defined in `common/config.json` before running system-level tests.

---

## License

This module follows the same license as the main repository.
