# Security System – API Service

This module implements the **API layer** of the Security System using **FastAPI**.

The API is responsible for receiving image frames captured by the camera of the self-service machine, validating the frames, and forwarding them to the appropriate detection models through the security system controller.

It acts as the communication bridge between the data acquisition layer (self-service machine cameras/clients) and the security system controller, which orchestrates the interaction with the inference services.

---

## Features

- Receives image frames captured from cameras
- Validates incoming frames (camera status and integrity)
- Forwards frames to different detection models via the system controller
- Supports multiple detection services:
  - Weapon detection 
  - Emotion detection
  - Violence Environment detection
- Asynchronous processing for improved performance and scalability
- Batch-based frame processing with buffer control
- REST API endpoints designed for integration with the self-service machine security system.

---

## Buffer Configuration

The API employs an internal buffering mechanism to support batch-based asynchronous processing.

The buffer size must be dimensioned based on the average inference cycle time and the incoming client frame rate (FPS), ensuring that the system can absorb incoming frames without overflow and maintain real-time operation.



## Installation

### 1. Create a virtual environment
```bash
python -m venv venv
```

### 2. Activate the virtual environment
- **Linux / macOS**
```bash
source venv/bin/activate
```

- **Windows (CMD / PowerShell)**
```bash
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Running the API

### Run directly with Python
```bash
python api.py
```

### Run with Uvicorn (recommended for development)
```bash
uvicorn api:app --host 0.0.0.0 --port 8050 --reload
```

---

## API Endpoints

### Health Check
```http
GET /
```

### Process Frame
```http
POST /process_frame
```

Example:
```bash
curl -X POST "http://127.0.0.1:8050/process_frame"      -H "Content-Type: multipart/form-data"      -F "frame=@/absolute/path/to/image.jpg"
```

### System Status
```http
GET /status
```

---

## Architecture Overview

- The API does **not perform inference directly**
- All frames are forwarded to the **Security System Controller**
- Each detection model runs as an **independent service with its own API**

---

## Building an Executable (Optional)

```bash
python -m PyInstaller --clean --noconfirm --name SecurityDetection   --onefile --paths ..   --hidden-import CONTROLADOR.controlador_batch   --hidden-import COMMON.common_batch   --hidden-import COMMON.logging_config   api.py
```

---

## License

This module follows the same license as the main Master's Thesis repository.
