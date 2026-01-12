# 🖥️ Benchmark & Performance Analysis — Hefesto Mini-PC

This directory contains performance benchmarking data and batch inference timing results generated directly from the Hefesto mini-PC during experimental testing. The files support empirical findings referenced in the thesis, specifically related to runtime performance and model throughput in real hardware conditions.

---

## 📁 Files

### `benchmark_results.xlsx`

> **Purpose:** Records system-level benchmarking results under varied model configurations and workloads.

#### 📌 Description

This Excel file includes key performance indicators captured during end-to-end execution of the detection system on the Hefesto device. It provides:

- Execution time per image
- Total processing time for the validation set
- CPU usage and inference times per model
- Memory footprint (if applicable)
- Comparative performance under different weight configurations

These metrics were crucial for quantifying the trade-offs between model accuracy and runtime efficiency, informing decisions on the optimal weight distribution in `config.json`.

---

### `BatchTempos.xlsx`

> **Purpose:** Captures detailed batch-level inference timing data across multiple models.

#### 📌 Description

This file includes granular logs of inference duration per model and per batch, allowing precise profiling of computational load. Each row typically corresponds to:

- Model name or batch ID
- Start and end timestamps
- Total duration per inference call

Used to identify bottlenecks and verify whether the system meets real-time processing requirements, particularly in resource-constrained edge environments.

---

## ⚙️ Context of Use

These two files were generated and analyzed during the experimental phase on the **Hefesto mini-computer**, a low-power edge device used to simulate deployment conditions. The data fed directly into:

- Evaluation of system scalability and latency
- Detection of performance regressions after weight reconfiguration
- Empirical justification for parameter tuning and scoring thresholds

Referenced in thesis sections discussing runtime evaluation, model latency, and hardware benchmarking.

---

## 📈 Outcomes

Analysis revealed that:

- Weight-adjusted configurations improved detection sensitivity without significant performance penalties
- Certain models had disproportionately high inference time and were deprioritized
- The system maintained acceptable latency on Hefesto under operational load

---

## 🔍 Next Steps (Optional)

- Automate data collection with timestamped logs via Python or shell scripts
- Visualize batch times over time to detect thermal throttling or performance drift
- Extend to GPU-based benchmarking (if applicable)

---

## 📌 Note

These results are hardware-dependent and reflect the specific performance characteristics of the Hefesto mini-PC. Use with caution when extrapolating to other environments.

---
