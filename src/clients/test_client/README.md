# Security System – Test Client

This directory contains **client applications used for testing and evaluation** of the Security System.

These clients are **not part of the production system**. They are used to simulate a self-service machine camera and validate the end-to-end behavior of the system.

---

## Purpose

The test client is responsible for:
- Capturing frames from a local camera (webcam)
- Sending image frames to the Security System API
- Controlling the frame rate (FPS)
- Displaying system responses and logs
- Querying the system alarm status
- Supporting performance evaluation and debugging

---

## Available Clients

- `client.py` – Initial prototype client
- `clientV2.py` – **Main GUI-based test client**
- `enviar_frames.py` – Simple script for sending images manually

---

## Prerequisites

- Python 3.x
- A camera/webcam
- A running instance of the Security System API

The client must communicate with the Security System API, which must be running and reachable at a known host and port (e.g., `http://127.0.0.1:8050`).

---

## ClientV2 – GUI Frame Sender

`clientV2.py` is the main testing client and provides a graphical user interface (GUI) for interacting with the Security System.

### Features

- Live camera capture using OpenCV
- Configurable FPS (frames per second)
- Asynchronous frame transmission to the API
- Real-time logging of:
  - API responses
  - Alarm decisions
  - Processing time
  - Detection results
- Manual query of the system status (`/status`)
- Optional export of processing times to Excel
- Non-blocking operation using `asyncio` and `aiohttp`

---

## How It Works

1. Frames are captured from the local camera
2. Each frame is encoded as JPEG
3. Frames are sent individually to the API endpoint:
   ```
   POST /process_frame
   ```
4. The API buffers frames internally and processes them in batch mode
5. The client receives and displays:
   - Alarm decision
   - Processing time
   - Detection labels
6. The client may query the last alarm state using:
   ```
   GET /status
   ```

---

## Running the ClientV2

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the client:

```bash
python clientV2.py --port 8050
```

Where:
- `8050` is the port where the Security System API is running

---

## Notes

- Each request sends **one frame**
- Frames are buffered internally by the API
- The FPS should be adjusted according to system processing capacity
- This client is intended for testing and experimental evaluation only

---

## Academic Context

These client applications were developed by **Vasco Maria** as part of a Master's Thesis to evaluate the behavior, performance, and robustness of the Security System under different operating conditions.

---

## License

This client follows the same license as the main repository.
