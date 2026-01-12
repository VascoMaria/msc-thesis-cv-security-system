# Security System – Controller Service

This module implements the **Security System Controller**, responsible for orchestrating communication between the API layer and the distributed inference services.

The controller does **not perform inference itself**. Instead, it coordinates requests to multiple detection services, aggregates their results, applies decision rules, and produces a final alarm decision.

---

## Responsibilities

- Receives batches of image frames from the API layer
- Dispatches frames to multiple inference services asynchronously
- Supports multiple detection categories, such as:
  - Weapon detection
  - Emotion recognition
  - Environment analysis
- Aggregates model responses per category
- Applies category-level and global decision rules
- Produces a final alarm decision and relevant detection labels
- Returns structured results to the API layer

---

## Architecture Role

The controller acts as the **central decision-making unit** of the security system.

System flow:

1. The API receives image frames from the self-service machine camera
2. Frames are forwarded to the controller in batch mode
3. The controller sends one request per model to each active inference service
4. Model results are aggregated by category
5. Decision rules are applied:
   - Per-category decision
   - Global alarm decision
6. The final response is returned to the API

This design enables:
- Modular and distributed inference services
- Independent scaling of models
- Clear separation between inference and decision logic

---

## Decision Logic

### 1. Category-Level Decision
For each active category, the controller:
- Collects all model outputs for that category
- Applies a category-specific decision rule
- Produces a boolean alarm decision per category

### 2. Global Decision
The final alarm decision is computed using one of the following strategies:
- **Default rule**: alarm if any category triggers
- **Scoring-based rule**: weighted sum of category alarms compared to a threshold
- **Consensus rule**: alarm if the majority of categories trigger

The active rule, thresholds, and weights are configurable via the `common` module.

---

## Additional Contextual Analysis

In addition to alarm decisions, the controller may extract **extra contextual information**, such as:

- Relative face area ratios between detected individuals
- Proximity indicators (e.g., presence of a second face near the user)

These flags provide supplementary information and do not directly affect the alarm decision unless explicitly configured.

---

## Implementation Details

- Asynchronous communication using `asyncio` and `aiohttp`
- One HTTP request per model per batch
- Shared HTTP session for efficiency
- Batch-based processing to reduce overhead
- Detailed logging for debugging and traceability

---

## Configuration

The controller relies on configuration and utility functions provided by the `common` module, including:

- Active detection categories
- Model endpoints per category
- Decision rules and thresholds
- Category weights
- Alarm-relevant labels

This allows the system behavior to be adjusted without modifying controller logic.

---

## Integration Notes

- The controller is invoked exclusively by the API layer
- Inference services must be running and accessible via HTTP
- Each inference service exposes its own endpoint
- The controller assumes consistent response formats from models

---

## Outputs

The controller returns a structured response containing:
- Overall processing time
- Alarm status (true/false)
- Alarm-related detection labels
- Raw model results
- Optional contextual flags and technical details

---

## Logging

- Informational logs for normal operation
- Warning-level logs when an alarm is detected
- Debug logs for detailed inspection of intermediate results

---

## License

This module is part of the Master's Thesis project and follows the same license as the main repository.
