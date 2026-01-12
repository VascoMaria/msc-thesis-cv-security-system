# Security System – Source Code

This directory contains the complete **source code** developed as part of **my Master's Thesis** by **Vasco Maria**.

The project implements a **modular and distributed security system** designed for real-time monitoring in a **self-service machine environment**, integrating multiple computer vision models, decision fusion logic, and experimental evaluation components.

---

## Overview

The system processes image frames captured by a self-service machine camera and determines whether a security alarm should be triggered.

The architecture separates:
- Core system logic
- Client-side data acquisition
- Experimental evaluation and benchmarking

This separation ensures clarity, reproducibility, and extensibility.

---

## Directory Structure

```text
src/
├─ security_system/   # Core security system implementation
│  ├─ api/            # FastAPI service (system entry point)
│  ├─ controller/     # Central orchestration and decision logic
│  ├─ inference_engine/ # Distributed inference services (one per model)
│  └─ common/         # Shared configuration, decision rules, utilities
│
├─ clients/           # Test and simulation clients
│  └─ test_client/    # Camera-based client simulating a self-service machine
│
└─ experiments/       # Experimental evaluation and benchmarking
   ├─ models_evaluation/  # Individual model evaluation experiments
   └─ system_evaluation/  # End-to-end system-level experiments
```

Each major subdirectory contains its own `README.md` with detailed documentation.

---

## Core Components

### Security System (`security_system/`)
Implements the final security system architecture, including:
- Asynchronous frame ingestion via API
- Centralized decision-making logic
- Distributed inference services
- Configuration-driven behavior (models, thresholds, weights)

This directory represents the **production-level implementation** of the system.

---

### Clients (`clients/`)
Contains **test and simulation clients** used during development and evaluation.

The clients simulate the behavior of a self-service machine by:
- Capturing frames from a local camera
- Sending frames to the system API
- Controlling frame rate (FPS)
- Querying system alarm status

These clients are **not part of the production system**.

---

### Experiments (`experiments/`)
Contains all **experimental work** conducted during the Master's Thesis, including:
- Individual model performance evaluation
- Batch processing benchmarks
- System-level decision fusion analysis
- Threshold and weight tuning
- Spatial and contextual feature evaluation (e.g., user proximity)

This directory supports the quantitative analysis presented in the dissertation.

---

## Design Principles

- Modular and service-oriented architecture
- Clear separation between inference, decision, and acquisition layers
- Configuration-driven system behavior
- Asynchronous and batch-based processing
- Designed for real-time operation in self-service environments

---

## Scope and Exclusions

This directory contains **only source code and experimental scripts**.

The following elements are intentionally kept outside `src/`:
- Datasets
- Model training pipelines
- Large model weights
- Executable artifacts (distributed via releases or artifacts)

---

## Academic Context

This project was developed by **Vasco Maria** as part of a Master's Thesis focused on:
- Computer vision for security applications
- Distributed inference architectures
- Decision fusion across heterogeneous models
- Real-time monitoring in self-service machines

---

## License

This project follows the same license as the main repository.
