# How to Use the Security System (Executables)

This document explains how to run the **Security System using executable artifacts**.

⚠️ **The executable files are not stored directly in this repository.**  
They are distributed through the **GitHub Releases** section of this repository.

Please download the required executables from the appropriate release before following the instructions below.

This guide describes the correct startup order for the computer vision services, the security system, and the camera client.

## What’s included

### Computer Vision microservices (start these first)
Based on the executables in this folder:

- `weaponDetectionBest.exe`
- `weaponDetectionBest2.exe`
- `weaponDetectionModel.exe`
- `efficient-net.exe`
- `emotion_deepface.exe`
- `emotion_fer.exe`
- `violenceDetection.exe`

> These services expose HTTP endpoints (e.g., `/detect_batch`) that the SecurityDetection system calls.

### Helper script
- `run_models.bat` — starts all computer vision microservices automatically (recommended).

### Security system (start after microservices)
- `SecurityDetection.exe` *(or similarly named executable in your build)*

### Client
- `Frame Streamer` — camera client that captures frames and sends them to the SecurityDetection system.

---

## 1) Start the Computer Vision microservices

### Option A — Start all services automatically (recommended)
1. Double-click **`run_models.bat`** (or run it from a terminal).
2. Keep all opened windows running.

### Option B — Start services manually
Run each microservice executable one-by-one (keep them open):
- `weaponDetectionBest.exe`
- `weaponDetectionBest2.exe`
- `weaponDetectionModel.exe`
- `efficient-net.exe`
- `emotion_deepface.exe`
- `emotion_fer.exe`
- `violenceDetection.exe`

---

## 2) Configure the SecurityDetection system

The SecurityDetection system reads a configuration file that contains the **ports/URLs** of every microservice.

### Config file
- `config.json` (example location: `./config.json`)

Make sure that the URLs in `config.json` match the ports your executables are actually using.

### Default microservice URLs and ports (from `config.json`)
| Category | Model/Service | URL | Port |
|---|---|---|---|
| weapon_detection | best | `http://localhost:8000/detect_batch` | `8000` |
| weapon_detection | best2 | `http://localhost:8007/detect_batch` | `8007` |
| weapon_detection | model | `http://localhost:8006/detect_batch` | `8006` |
| weapon_detection | EffientNet | `http://localhost:8010/detect_batch` | `8010` |
| emotion_recognition | deepface | `http://localhost:8001/detect_batch` | `8001` |
| emotion_recognition | fer | `http://localhost:8012/detect_batch` | `8012` |
| environment_detection | violenceEnvironment_detection | `http://localhost:8009/detect_batch` | `8009` |

> If you change any port, **update the corresponding URL** in `config.json`.

---

## 3) Start the SecurityDetection system

After **all microservices are running** and `config.json` is updated:

1. Run **`SecurityDetection.exe`**
2. Confirm it loads `config.json` successfully (check its console logs).

---

## 4) Start the Frame Streamer client

Finally, start the client that captures frames from a camera and sends them to the SecurityDetection system:

1. Run **Frame Streamer**

2. Start streaming.

> The exact endpoint/path depends on how your SecurityDetection server is implemented (check its console output or documentation).

---

## Troubleshooting

- **A service won’t start / “port already in use”**  
  Another process is using the same port. Stop the other process or change the service port and update `config.json`.

- **SecurityDetection returns errors calling a model**  
  Verify:
  1) the microservice window is still running,  
  2) the URL in `config.json` matches the service port, and  
  3) the service endpoint is reachable in a browser (e.g., open the base host and confirm it responds).

- **Frame Streamer shows connection errors**  
  Ensure SecurityDetection is running and that Frame Streamer is pointing to the correct host/port/endpoint.

---

## Notes
- Start order matters:
  1) **Computer Vision microservices** (via `.bat` or manually)  
  2) **SecurityDetection**  
  3) **Frame Streamer**
