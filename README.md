# Computer Vision–Based Security System for Self-Service Machines

This repository contains the complete work developed as part of a **Master’s Thesis in Software Engineering**  
**“Visão Computacional para Soluções de Segurança Adicional em Máquinas de Autoatendimento”**  
by **Vasco Miguel Raimão Maria**, submitted to the **Faculdade de Ciências da Universidade de Lisboa**.

This work was developed during a professional internship at **INM – Innovation Makers**, a technology company, and focuses on the design, implementation, and evaluation of an additional computer vision–based security layer for the banking sector, specifically targeting self-service machines such as ATMs and VTMs.


---

## Project Overview

Self-service machines are widely used and play a critical role in modern service delivery. However, they are increasingly exposed to physical security threats such as assaults, coercion, vandalism, and suspicious behavior around the user.

This project addresses these challenges by introducing a **computer vision–based security system** that operates as an **independent and complementary security layer**, without requiring modifications to the core software of the self-service machine.

The system analyzes image frames captured by the machine’s camera and detects potentially dangerous situations in real time.

---

## System Objectives

The main objectives of the proposed system are:

- To enhance the physical security of self-service machines
- To detect potentially dangerous situations using visual information
- To integrate multiple computer vision models in a modular architecture
- To support real-time operation under constrained computational resources
- To provide configurable and explainable alarm decision logic

---

## System Capabilities

The security system integrates multiple computer vision components to detect:

- **Weapons**, including firearms and small bladed weapons
- **Facial emotions** associated with risk scenarios (e.g., fear, aggression)
- **Violent or suspicious environments**
- **User proximity**, modeling violations of personal space around the machine

The outputs of these models are combined through **configurable decision rules**, thresholds, and weights to determine whether a security alarm should be triggered.

---

## Repository Structure

```text
.
├─ src/        # Source code of the security system, clients, and experiments
├─ thesis/     # Master's thesis (PDF and LaTeX source)
└─ README.md   # This file

### `src/`

Contains the complete implementation of the system, including:
- The core security system (API, controller, inference services, common)
- Test and simulation clients
- Experimental evaluation and benchmarking

Each subdirectory includes its own documentation.

---

### `thesis/`

Contains the Master’s Thesis document and its LaTeX source files, documenting:
- System architecture and design decisions
- Model selection and evaluation
- Experimental validation and performance analysis

---

## Architecture Highlights

The proposed system follows a **modular and distributed architecture**, designed to operate as an independent security layer for self-service machines.

Key architectural characteristics include:
- Modular and service-oriented design
- Distributed inference services (one service per model)
- Centralized decision controller responsible for alarm logic
- Configuration-driven behavior (models, thresholds, weights, rules)
- Asynchronous and batch-based processing
- Designed for real-time operation under constrained computational resources

This architecture enables flexibility, scalability, and easy integration with different machine configurations.

---

## Experiments and Validation

This repository includes extensive experimental work conducted to validate the proposed system.

The experimental evaluation covers:
- Individual model evaluation and comparison
- System-level performance analysis
- Batch processing benchmarks
- Threshold and weight tuning
- Spatial and contextual feature analysis, such as user proximity and personal space violations

The experimental results demonstrate the feasibility of the proposed approach and support its applicability in real-world deployment scenarios.

---

## Executable Artifacts (Planned)

A dedicated directory containing **compiled executables** will be added in a future update.

These executables will include:
- The main security system
- Individual inference services
- Test client applications

The executable artifacts are intended for deployment and demonstration purposes and will be distributed separately from the source code.

---

## Academic and Industrial Context

This work was developed during a **professional internship at INM – Innovation Makers**, within the scope of applied research on intelligent security systems.

The project contributes to research topics such as:
- Computer vision for security applications
- Distributed inference architectures
- Decision fusion across heterogeneous models
- Real-time monitoring in self-service environments

---

## License

This repository is intended for **academic and research purposes**.

License details are provided in the repository root.
