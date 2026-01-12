
# ⚙️ Model Configuration Evaluation Pipeline

This repository contains the core scripts and output files used in the experimental evaluation of model weighting adjustments in a multi-model decision system. These adjustments are part of a decision logic tuning process based on performance-driven scoring strategies.

The material herein directly supports the thesis section _"Ajuste dos Pesos no Ficheiro de Configuração"_.

---

## 📌 Context & Objective

The system under evaluation employs multiple AI/ML models whose outputs are aggregated through a weighted scoring mechanism. Initially, all models contributed equally. This repository includes the tooling used to reassess and empirically calibrate those weights based on real-world validation performance, feature importance metrics, and individual model recall rates.

The primary goal is to ensure higher sensitivity (recall) in risk detection while maintaining an optimal balance in false positive and false negative rates.

---

## 📁 Contents

### `metricas_config.py`

> **Purpose:** Automated computation of classification metrics from system output under a specific configuration of model weights.

#### 🔍 Description

This script ingests evaluation results (from `resultado_avaliacao_comConfig.xlsx`), validates the input, and calculates a set of performance metrics:

- **Confusion matrix elements**: TP, FP, TN, FN  
- **Performance indicators**:  
  - Accuracy  
  - Precision  
  - Recall (Sensitivity)  
  - F1-Score  
  - False Negative Rate (FNR)

The computed metrics are then exported to `metricas_avaliacao.xlsx`, supporting longitudinal comparison across multiple configuration runs.

#### 📦 Dependencies

```bash
pandas
openpyxl
scikit-learn
```

#### 🧠 Logic Highlights

- Binary conversion of labels (ground truth and predictions).
- Runtime validation for missing or malformed data.
- If `metricas_avaliacao.xlsx` exists, results are appended as a new sheet.
- Otherwise, the file is initialized with the first metrics batch.

---

### `metricas_avaliacao.xlsx`

> **Purpose:** Aggregated metric reports across different model weighting configurations.

This Excel file is automatically updated by `metricas_config.py` and includes side-by-side comparisons of:

- **Baseline Configuration** – All model weights equally distributed.
- **Adjusted Configuration** – Weights derived from:
  - Mean feature importances from Random Forest analysis
  - Model-level recall performance from prior evaluations
  - Detection thresholds (`thresholdDetection`) calibrated via confidence scores and false negative patterns

#### 📈 Use in Thesis

This file serves as the empirical foundation for the comparative analysis presented in:

- `Figura: Comparação entre Configurações`  
- Subsection: _Ajuste dos Pesos no Ficheiro de Configuração_

---

## ✅ Outcomes

Following the reconfiguration:

- **Recall increased**, leading to better detection of critical risk events.
- **FNR decreased**, reducing missed detections.
- **F1-Score remained stable**, ensuring precision wasn't sacrificed.

This demonstrates a net positive shift in system behavior under real validation scenarios, justifying the scoring logic update in the `config.json`.

---

## 🧩 Future Improvements

- Integrate a config diff comparator to quantify impact per parameter.
- Automate threshold tuning via Bayesian Optimization.
- Include per-model confusion matrices for deeper insights.

---

## 👨‍💻 Maintainer

This project was developed and maintained as part of an academic-industry collaboration focused on intelligent threat detection systems.

---
