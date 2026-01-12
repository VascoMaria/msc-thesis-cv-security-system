# Models Evaluation Experiments

This directory contains experiments focused on the **individual evaluation of computer vision models** used in the Security System.

These experiments assess the predictive performance of each model in isolation, without considering system-level decision fusion. The evaluation results stored in `model_evaluation_metrics.xlsx` were obtained through controlled testing on a validation dataset and are used to guide system configuration and model prioritization.

---


# Models Evaluation Experiments

This directory contains experiments focused on the **individual evaluation of computer vision models** used in the Security System.

The goal of these experiments is to assess the performance of each model independently, without considering system-level decision fusion.

---

## Scope

The experiments in this directory focus on:
- Accuracy and detection performance of individual models
- Confidence score analysis
- Class-level detection behavior
- Model comparison under the same input conditions

These results are used to:
- Compare different models within the same category
- Select candidate models for system integration
- Support quantitative analysis presented in the Master's Thesis

---

## Contents

- `model_evaluation_metrics.xlsx`  
  Aggregated evaluation metrics for all tested models (e.g., precision, recall, confidence scores).

- `Screenshot_14.png`  
  Visual reference related to the evaluation process or results.

- `README.md`  
  This documentation file.

---

## Notes

- These experiments evaluate **models in isolation**
- No system-level decision rules or weights are applied here
- Results from this directory feed into the system-level evaluations

---

## Academic Context

These experiments were conducted by **Vasco Maria** as part of a Master's Thesis to evaluate and compare individual computer vision models prior to their integration into a distributed security system.



## 📁 File: `model_evaluation_metrics.xlsx`

### 🎯 Purpose

This Excel file contains **unified evaluation metrics** across all active models in the system. It serves as the main source of truth for comparing model effectiveness based on classification metrics.

---

### 📌 Contents

Each row in the spreadsheet corresponds to an evaluated model and includes the following metrics:

- **True Positives (TP)**
- **True Negatives (TN)**
- **False Positives (FP)**
- **False Negatives (FN)**
- **Accuracy**
- **Precision**
- **Recall (Sensitivity)**
- **F1-Score**
- **False Negative Rate (FNR)**

---

### 🧠 Use Cases

- Identify the most reliable models for threat detection based on recall and FNR.
- Support configuration of decision logic (`config.json`) using empirical model performance.
- Provide traceable, quantitative justification for model selection in production.

---

### 📈 Interpretation

- **High Recall** indicates the model effectively detects true risk events.
- **Low FNR** suggests the model minimizes missed detections.
- **Balanced F1-score** supports confidence in both precision and sensitivity.

These values directly informed scoring weight adjustments and threshold tuning during system calibration.

---

## 🧪 Evaluation Conditions

- All results were generated using the final validation dataset.
- Execution and metric collection were performed on the Hefesto mini-PC under real deployment conditions.
- Evaluation was part of the model selection and system finalization phase.

---

## 📌 Notes

Ensure consistency in evaluation methodology when comparing new models against this benchmark. For any additions, update this file to preserve unified reporting.

---
