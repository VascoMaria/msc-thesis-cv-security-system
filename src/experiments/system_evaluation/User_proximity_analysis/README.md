# User Proximity Analysis

This directory contains experiments related to the **user proximity feature**, an additional spatial feature designed to detect potential **personal space violations** around a self-service machine.

This feature models how close another person is to the primary user of the machine and is used to enhance the overall security assessment of the system.

---

## Purpose

The goal of the user proximity analysis is to:
- Quantify the distance between detected individuals and the machine user
- Identify situations where another person enters the user's personal space
- Define and validate proximity thresholds that indicate potentially risky behavior
- Support decision-making at the system level through an additional contextual feature

---

## Feature Description

The user proximity feature is computed using **region-based spatial measurements** derived from detected face or body areas.

It captures:
- Relative distance ratios between detected regions
- Area-based proximity indicators
- Normalized spatial relationships independent of camera resolution

This feature is evaluated independently before being integrated into system-level decision rules.

---

## Experiments and Analysis

The experiments in this directory include:
- Proximity metric computation
- Distribution analysis of proximity values
- Threshold validation to determine personal space boundaries
- ROC and confidence-based analysis (when applicable)
- Visual inspection of proximity-related cases

---

## Threshold Validation

A key objective of this analysis is to determine an appropriate **proximity threshold** that defines when a personal space violation occurs.

Thresholds are selected based on:
- Statistical distribution of proximity values
- Validation dataset behavior
- Trade-off between sensitivity and false positives

The selected threshold is later used in the system configuration and decision rules.

---

## Notes

- This analysis is conducted **offline** and independently from the real-time system
- Results from this directory inform system-level configuration parameters
- The proximity feature complements other computer vision models and does not perform detection on its own

---

## Academic Context

This analysis was developed by **Vasco Maria** as part of a Master's Thesis to enhance security assessment by incorporating spatial context and personal space awareness into a distributed computer vision system for self-service machines.

---

## License

This directory follows the same license as the main repository.
