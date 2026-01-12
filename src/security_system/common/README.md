# Security System – Common Module

This module implements the **shared configuration, rules, and utility logic** used across the Security System.

It centralizes all configuration loading, decision rules, thresholds, weights, and helper functions that are required by both the API layer and the Controller service.

The goal of this module is to ensure **consistent behavior**, **config-driven decisions**, and **separation of concerns** across the system.

---

## Responsibilities

- Load and cache system configuration from `config.json`
- Provide access to active detection categories and models
- Store and expose thresholds, weights, and alarm-related classes
- Implement category-level decision rules
- Support global alarm decision logic
- Provide utility functions shared across modules
- Perform basic camera frame validation

---

## Configuration Management

### Configuration File

The module loads its configuration from a `config.json` file.

The configuration file is resolved dynamically:
- When running as a **PyInstaller executable**, the config is loaded from the executable directory
- When running as Python code, the config is loaded from the module directory

The configuration is loaded **once** and cached in memory for performance.

---

## Cached Configuration Access

To avoid repeated file access, the configuration is cached using an in-memory mechanism.

The following elements are extracted and stored:
- Active categories
- Models per category
- Alarm-relevant classes per model
- Detection thresholds
- Model and category weights

This allows fast access during real-time processing.

---

## Detection Categories and Models

The module provides helper functions to:
- Retrieve all active detection categories
- Retrieve model endpoints grouped by category
- Retrieve per-model thresholds and weights
- Retrieve alarm-relevant labels per category

All values are **fully driven by configuration**, allowing behavior changes without modifying code.

---

## Decision Rules

### Category-Level Decision

For each detection category, the module supports multiple decision strategies:

- **Scoring**  
  A weighted sum of model detections is compared against a threshold.

- **Consensus**  
  An alarm is triggered if the majority of models detect an alarm condition.

- **Priority**  
  A class-weighted strategy where specific alarm classes may immediately trigger an alarm.

The active rule per category is defined in the configuration file.

---

### Global Decision Logic

In addition to category-level rules, the module supports **global alarm strategies**, such as:
- Default (any category triggers)
- Scoring-based aggregation
- Consensus across categories

Thresholds and weights are configurable per category.

---

## Alarm Label Filtering

The module builds and caches a set of **alarm-relevant labels** per category.

Only detections matching these labels are considered when:
- Computing alarms
- Extracting final alarm labels
- Producing alarm summaries

This prevents non-relevant detections from influencing decisions.

---

## Additional Contextual Analysis

The module supports optional contextual analysis, such as:
- Face area ratio thresholds
- Proximity indicators between detected individuals

These values are configurable and may be used by the Controller for additional context, without directly affecting alarm logic unless configured.

---

## Camera Frame Validation

A basic camera validation function is provided to:
- Detect blocked cameras
- Detect excessive brightness
- Validate normal camera operation

This helps discard invalid frames early in the pipeline.

---

## Design Principles

- Configuration-driven behavior
- Single source of truth for rules and thresholds
- Shared logic across all system components
- Optimized for real-time and batch processing
- Explicit separation between configuration, decision logic, and inference

---

## Integration Notes

- This module is imported by both the API and the Controller
- It must be available in all runtime environments
- Changes to `config.json` affect system behavior globally

---

## License

This module is part of the Master's Thesis project and follows the same license as the main repository.
