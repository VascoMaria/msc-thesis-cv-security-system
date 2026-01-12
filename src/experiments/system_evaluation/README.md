# System Evaluation Experiments

This directory contains **system-level experiments** conducted on the complete Security System.

Unlike individual model evaluations, these experiments analyze the behavior of the **full distributed system**, including decision fusion, weighting strategies, batch processing, and the integration of additional contextual features.

---

## Scope

The experiments in this directory focus on:
- End-to-end system behavior
- Performance under batch-based processing
- Decision fusion across multiple inference services
- Impact of weights and thresholds on alarm decisions
- Integration of additional contextual and spatial features
- Comparison with baseline approaches

---

## Subdirectories

- `batch_benchmark_hefesto/`  
  Benchmarks related to batch processing performance, conducted on the Hefesto platform.

- `distribution_analysis/`  
  Analysis of score, class, and decision distributions produced by the system.

- `random_forest_experiments/`  
  Baseline experiments using Random Forest models for comparison with the proposed system.

- `weighted_system_evaluation/`  
  Experiments evaluating the impact of adjusted weights and decision rules on system performance.

- `user_proximity_analysis/`  
  Evaluation of an additional spatial feature modeling **user proximity and personal space violation**, including proximity metrics, distribution analysis, and threshold validation.

---

## Notes

- These experiments evaluate the **complete system pipeline**
- All active inference services, contextual features, and decision rules are enabled
- Results reflect realistic operating conditions of the security system

---

## Academic Context

These experiments were conducted by **Vasco Maria** as part of a Master's Thesis to validate the effectiveness, robustness, and performance of the proposed Security System architecture, including system-level decision fusion and contextual feature integration.
